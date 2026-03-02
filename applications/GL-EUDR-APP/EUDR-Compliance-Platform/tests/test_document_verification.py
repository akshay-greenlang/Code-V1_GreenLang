"""
Unit tests for GL-EUDR-APP v1.0 Document Verification Engine.

Tests document upload, classification, text extraction, verification,
EUDR article compliance checks, entity linking, and gap analysis.

Test count target: 30+ tests
"""

import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Document Verification Engine (self-contained for testing)
# ---------------------------------------------------------------------------

DOC_TYPES = {"CERTIFICATE", "PERMIT", "LAND_TITLE", "INVOICE", "TRANSPORT", "OTHER"}
VERIFICATION_STATUSES = {"pending", "verified", "failed", "expired"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


class DocVerificationError(Exception):
    pass


class DocumentNotFoundError(DocVerificationError):
    pass


class DocumentVerificationEngine:
    """Engine for document upload, classification, verification, and compliance."""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def upload_document(self, name: str, doc_type: str,
                        file_size_bytes: int = 0,
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not name or not name.strip():
            raise DocVerificationError("Document name is required")
        if doc_type not in DOC_TYPES:
            raise DocVerificationError(f"Invalid doc_type '{doc_type}'")
        if file_size_bytes > MAX_FILE_SIZE_BYTES:
            raise DocVerificationError(f"File exceeds max size of {MAX_FILE_SIZE_BYTES} bytes")
        if file_size_bytes < 0:
            raise DocVerificationError("File size cannot be negative")

        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)
        doc = {
            "document_id": doc_id,
            "name": name.strip(),
            "doc_type": doc_type,
            "file_size_bytes": file_size_bytes,
            "verification_status": "pending",
            "verification_score": None,
            "ocr_text": None,
            "compliance_findings": [],
            "linked_supplier_id": None,
            "linked_plot_id": None,
            "linked_dds_id": None,
            "metadata": metadata or {},
            "created_at": now,
            "updated_at": now,
        }
        self._store[doc_id] = doc
        return doc

    def classify_document(self, text_content: str) -> str:
        """Classify document type from extracted text content."""
        text_lower = text_content.lower()
        if any(kw in text_lower for kw in ["certificate", "certified", "certification"]):
            return "CERTIFICATE"
        if any(kw in text_lower for kw in ["invoice", "bill", "payment"]):
            return "INVOICE"
        if any(kw in text_lower for kw in ["permit", "license", "authorization"]):
            return "PERMIT"
        if any(kw in text_lower for kw in ["land title", "deed", "property"]):
            return "LAND_TITLE"
        if any(kw in text_lower for kw in ["transport", "shipping", "waybill", "bill of lading"]):
            return "TRANSPORT"
        return "OTHER"

    def extract_text(self, doc_id: str) -> str:
        """Simulate OCR text extraction."""
        doc = self._store.get(doc_id)
        if not doc:
            raise DocumentNotFoundError(f"Document '{doc_id}' not found")
        if doc.get("file_size_bytes", 0) == 0:
            return ""
        # Simulated OCR based on doc_type
        type_texts = {
            "CERTIFICATE": "This is a certified deforestation-free certificate issued by authority.",
            "PERMIT": "Export permit authorization for timber products.",
            "INVOICE": "Invoice for 500 tonnes of soy meal.",
            "LAND_TITLE": "Land title deed for property in Para, Brazil.",
            "TRANSPORT": "Bill of lading for maritime transport.",
            "OTHER": "General document content.",
        }
        text = type_texts.get(doc["doc_type"], "Unknown document.")
        doc["ocr_text"] = text
        return text

    def verify_document(self, doc_id: str) -> Dict[str, Any]:
        """Verify document completeness and validity."""
        doc = self._store.get(doc_id)
        if not doc:
            raise DocumentNotFoundError(f"Document '{doc_id}' not found")

        issues = []
        score = 1.0

        # Check required fields
        if not doc.get("name"):
            issues.append("Missing document name")
            score -= 0.3
        if not doc.get("doc_type"):
            issues.append("Missing document type")
            score -= 0.3

        # Check metadata for expiry
        expiry = doc.get("metadata", {}).get("expiry_date")
        if expiry:
            try:
                exp_date = datetime.fromisoformat(expiry)
                if exp_date.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
                    issues.append("Document has expired")
                    score -= 0.5
                    doc["verification_status"] = "expired"
            except (ValueError, TypeError):
                issues.append("Invalid expiry date format")
                score -= 0.2

        # Check file size (empty docs are suspicious)
        if doc.get("file_size_bytes", 0) == 0:
            issues.append("Document has no content (0 bytes)")
            score -= 0.2

        score = max(0.0, min(1.0, score))
        if not issues:
            doc["verification_status"] = "verified"
        elif doc["verification_status"] != "expired":
            doc["verification_status"] = "failed" if score < 0.5 else "pending"

        doc["verification_score"] = score
        doc["updated_at"] = datetime.now(timezone.utc)

        return {
            "document_id": doc_id,
            "verification_status": doc["verification_status"],
            "score": score,
            "issues": issues,
        }

    def check_compliance(self, doc_id: str) -> Dict[str, Any]:
        """Check EUDR article compliance for a document."""
        doc = self._store.get(doc_id)
        if not doc:
            raise DocumentNotFoundError(f"Document '{doc_id}' not found")

        articles = {}

        # Article 3: Deforestation-free
        ocr = (doc.get("ocr_text") or "").lower()
        articles["article_3"] = {
            "name": "Deforestation-free (Art. 3)",
            "compliant": "deforestation-free" in ocr or "deforestation free" in ocr,
            "details": "Product must be deforestation-free per EUDR cutoff date.",
        }

        # Article 4: Legality
        articles["article_4"] = {
            "name": "Legality (Art. 4)",
            "compliant": doc["doc_type"] in ("PERMIT", "CERTIFICATE", "LAND_TITLE"),
            "details": "Legal compliance documentation must be present.",
        }

        # Article 10: Geolocation
        has_geo = bool(doc.get("linked_plot_id"))
        articles["article_10"] = {
            "name": "Geolocation (Art. 10)",
            "compliant": has_geo,
            "details": "Document must be linked to geolocated production plot.",
        }

        # Article 11: Risk assessment
        has_risk = "risk" in ocr or doc.get("metadata", {}).get("risk_assessed", False)
        articles["article_11"] = {
            "name": "Risk Assessment (Art. 11)",
            "compliant": has_risk,
            "details": "Risk assessment must be conducted and documented.",
        }

        all_pass = all(a["compliant"] for a in articles.values())
        failed = [k for k, v in articles.items() if not v["compliant"]]

        return {
            "document_id": doc_id,
            "all_compliant": all_pass,
            "articles": articles,
            "failed_articles": failed,
        }

    def link_document(self, doc_id: str, entity_type: str,
                      entity_id: str) -> Dict[str, Any]:
        """Link a document to a supplier, plot, or DDS."""
        doc = self._store.get(doc_id)
        if not doc:
            raise DocumentNotFoundError(f"Document '{doc_id}' not found")
        if entity_type == "supplier":
            doc["linked_supplier_id"] = entity_id
        elif entity_type == "plot":
            doc["linked_plot_id"] = entity_id
        elif entity_type == "dds":
            doc["linked_dds_id"] = entity_id
        else:
            raise DocVerificationError(f"Invalid entity_type '{entity_type}'")
        doc["updated_at"] = datetime.now(timezone.utc)
        return doc

    def gap_analysis(self, supplier_id: str,
                     required_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze document gaps for a supplier."""
        if required_types is None:
            required_types = ["CERTIFICATE", "PERMIT", "LAND_TITLE"]

        supplier_docs = [
            d for d in self._store.values()
            if d.get("linked_supplier_id") == supplier_id
        ]
        present_types = {d["doc_type"] for d in supplier_docs}
        missing = [t for t in required_types if t not in present_types]

        severity = "low"
        if len(missing) >= 2:
            severity = "high"
        elif len(missing) == 1:
            severity = "medium"

        recommendations = []
        for m in missing:
            recommendations.append(f"Upload {m} document for supplier {supplier_id}")

        return {
            "supplier_id": supplier_id,
            "required_types": required_types,
            "present_types": list(present_types),
            "missing_types": missing,
            "severity": severity,
            "recommendations": recommendations,
            "complete": len(missing) == 0,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return DocumentVerificationEngine()


@pytest.fixture
def sample_doc(engine):
    return engine.upload_document("Forest Cert.pdf", "CERTIFICATE", file_size_bytes=1024)


# ---------------------------------------------------------------------------
# TestUploadDocument
# ---------------------------------------------------------------------------

class TestUploadDocument:

    def test_valid_upload(self, engine):
        doc = engine.upload_document("test.pdf", "CERTIFICATE", file_size_bytes=500)
        assert doc["document_id"].startswith("doc_")
        assert doc["verification_status"] == "pending"

    def test_validate_metadata(self, engine):
        doc = engine.upload_document("test.pdf", "CERTIFICATE",
                                     metadata={"issuer": "Bureau Veritas"})
        assert doc["metadata"]["issuer"] == "Bureau Veritas"

    def test_file_size_limit(self, engine):
        with pytest.raises(DocVerificationError, match="max size"):
            engine.upload_document("huge.pdf", "OTHER",
                                   file_size_bytes=100 * 1024 * 1024)

    def test_negative_file_size(self, engine):
        with pytest.raises(DocVerificationError, match="negative"):
            engine.upload_document("bad.pdf", "OTHER", file_size_bytes=-1)

    def test_empty_name_rejected(self, engine):
        with pytest.raises(DocVerificationError, match="name is required"):
            engine.upload_document("", "CERTIFICATE")

    def test_invalid_doc_type(self, engine):
        with pytest.raises(DocVerificationError, match="Invalid doc_type"):
            engine.upload_document("test.pdf", "SPREADSHEET")


# ---------------------------------------------------------------------------
# TestClassifyDocument
# ---------------------------------------------------------------------------

class TestClassifyDocument:

    def test_certificate_detection(self, engine):
        assert engine.classify_document("This is a certification document") == "CERTIFICATE"

    def test_invoice_detection(self, engine):
        assert engine.classify_document("Payment invoice for goods") == "INVOICE"

    def test_permit_detection(self, engine):
        assert engine.classify_document("Export permit for timber") == "PERMIT"

    def test_land_title_detection(self, engine):
        assert engine.classify_document("Land title deed for property") == "LAND_TITLE"

    def test_transport_detection(self, engine):
        assert engine.classify_document("Bill of lading for shipment") == "TRANSPORT"

    def test_unknown_type(self, engine):
        assert engine.classify_document("Random text about nothing") == "OTHER"


# ---------------------------------------------------------------------------
# TestExtractText
# ---------------------------------------------------------------------------

class TestExtractText:

    def test_ocr_returns_text(self, engine, sample_doc):
        text = engine.extract_text(sample_doc["document_id"])
        assert "certificate" in text.lower()

    def test_empty_document_returns_empty(self, engine):
        doc = engine.upload_document("empty.pdf", "OTHER", file_size_bytes=0)
        text = engine.extract_text(doc["document_id"])
        assert text == ""

    def test_nonexistent_raises(self, engine):
        with pytest.raises(DocumentNotFoundError):
            engine.extract_text("doc_nonexistent")


# ---------------------------------------------------------------------------
# TestVerifyDocument
# ---------------------------------------------------------------------------

class TestVerifyDocument:

    def test_valid_document_passes(self, engine, sample_doc):
        result = engine.verify_document(sample_doc["document_id"])
        assert result["verification_status"] == "verified"
        assert result["score"] > 0.5

    def test_zero_byte_document_penalized(self, engine):
        doc = engine.upload_document("empty.pdf", "CERTIFICATE", file_size_bytes=0)
        result = engine.verify_document(doc["document_id"])
        assert "no content" in result["issues"][0]

    def test_expired_certificate_fails(self, engine):
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        doc = engine.upload_document("old.pdf", "CERTIFICATE",
                                     file_size_bytes=100,
                                     metadata={"expiry_date": yesterday})
        result = engine.verify_document(doc["document_id"])
        assert result["verification_status"] == "expired"
        assert "expired" in result["issues"][0].lower()

    def test_nonexistent_raises(self, engine):
        with pytest.raises(DocumentNotFoundError):
            engine.verify_document("doc_nonexistent")


# ---------------------------------------------------------------------------
# TestComplianceCheck
# ---------------------------------------------------------------------------

class TestComplianceCheck:

    def test_article_3_deforestation_free(self, engine, sample_doc):
        engine.extract_text(sample_doc["document_id"])
        result = engine.check_compliance(sample_doc["document_id"])
        assert result["articles"]["article_3"]["compliant"] is True

    def test_article_4_legality(self, engine, sample_doc):
        result = engine.check_compliance(sample_doc["document_id"])
        assert result["articles"]["article_4"]["compliant"] is True

    def test_article_10_geolocation_no_plot(self, engine, sample_doc):
        result = engine.check_compliance(sample_doc["document_id"])
        assert result["articles"]["article_10"]["compliant"] is False

    def test_article_10_geolocation_with_plot(self, engine, sample_doc):
        engine.link_document(sample_doc["document_id"], "plot", "plot_123")
        result = engine.check_compliance(sample_doc["document_id"])
        assert result["articles"]["article_10"]["compliant"] is True

    def test_article_11_risk_assessment(self, engine):
        doc = engine.upload_document("risk.pdf", "OTHER", file_size_bytes=100,
                                     metadata={"risk_assessed": True})
        result = engine.check_compliance(doc["document_id"])
        assert result["articles"]["article_11"]["compliant"] is True

    def test_all_articles_pass(self, engine):
        doc = engine.upload_document("complete.pdf", "CERTIFICATE", file_size_bytes=100,
                                     metadata={"risk_assessed": True})
        engine.link_document(doc["document_id"], "plot", "plot_abc")
        doc_record = engine._store[doc["document_id"]]
        doc_record["ocr_text"] = "deforestation-free certificate with risk assessment"
        result = engine.check_compliance(doc["document_id"])
        assert result["all_compliant"] is True

    def test_some_articles_fail(self, engine):
        doc = engine.upload_document("incomplete.pdf", "OTHER", file_size_bytes=100)
        result = engine.check_compliance(doc["document_id"])
        assert result["all_compliant"] is False
        assert len(result["failed_articles"]) > 0


# ---------------------------------------------------------------------------
# TestLinkDocument
# ---------------------------------------------------------------------------

class TestLinkDocument:

    def test_link_to_supplier(self, engine, sample_doc):
        result = engine.link_document(sample_doc["document_id"], "supplier", "sup_123")
        assert result["linked_supplier_id"] == "sup_123"

    def test_link_to_plot(self, engine, sample_doc):
        result = engine.link_document(sample_doc["document_id"], "plot", "plot_456")
        assert result["linked_plot_id"] == "plot_456"

    def test_link_to_dds(self, engine, sample_doc):
        result = engine.link_document(sample_doc["document_id"], "dds", "dds_789")
        assert result["linked_dds_id"] == "dds_789"

    def test_invalid_entity_type_raises(self, engine, sample_doc):
        with pytest.raises(DocVerificationError, match="Invalid entity_type"):
            engine.link_document(sample_doc["document_id"], "order", "ord_123")

    def test_link_nonexistent_doc_raises(self, engine):
        with pytest.raises(DocumentNotFoundError):
            engine.link_document("doc_nonexistent", "supplier", "sup_1")


# ---------------------------------------------------------------------------
# TestGapAnalysis
# ---------------------------------------------------------------------------

class TestGapAnalysis:

    def test_all_docs_present(self, engine):
        for dt in ["CERTIFICATE", "PERMIT", "LAND_TITLE"]:
            doc = engine.upload_document(f"{dt}.pdf", dt, file_size_bytes=100)
            engine.link_document(doc["document_id"], "supplier", "sup_full")
        result = engine.gap_analysis("sup_full")
        assert result["complete"] is True
        assert result["severity"] == "low"
        assert len(result["missing_types"]) == 0

    def test_missing_certificate(self, engine):
        for dt in ["PERMIT", "LAND_TITLE"]:
            doc = engine.upload_document(f"{dt}.pdf", dt, file_size_bytes=100)
            engine.link_document(doc["document_id"], "supplier", "sup_partial")
        result = engine.gap_analysis("sup_partial")
        assert "CERTIFICATE" in result["missing_types"]
        assert result["severity"] == "medium"

    def test_missing_multiple(self, engine):
        doc = engine.upload_document("cert.pdf", "CERTIFICATE", file_size_bytes=100)
        engine.link_document(doc["document_id"], "supplier", "sup_two_missing")
        result = engine.gap_analysis("sup_two_missing")
        assert len(result["missing_types"]) == 2
        assert result["severity"] == "high"

    def test_severity_scoring(self, engine):
        result = engine.gap_analysis("sup_none")
        assert result["severity"] == "high"  # all 3 missing

    def test_recommendations(self, engine):
        result = engine.gap_analysis("sup_empty")
        assert len(result["recommendations"]) == 3
        for rec in result["recommendations"]:
            assert "Upload" in rec
