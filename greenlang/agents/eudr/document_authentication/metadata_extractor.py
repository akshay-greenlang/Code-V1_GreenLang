# -*- coding: utf-8 -*-
"""
Metadata Extractor Engine - AGENT-EUDR-012 Engine 5

Extract and validate document metadata for authentication across all
EUDR supply chain document types. Covers PDF metadata (Title, Author,
Creator, Producer, CreationDate, ModDate, Keywords), EXIF metadata
from embedded images (GPS, camera model, capture date), XMP metadata,
serial/reference number extraction, and cross-validation of claimed
attributes against extracted metadata for anomaly detection.

Zero-Hallucination Guarantees:
    - All metadata extraction uses deterministic parsing (no ML/LLM)
    - Date comparison uses Python datetime arithmetic
    - String matching uses deterministic Levenshtein distance / exact match
    - GPS coordinate extraction is pure numeric parsing
    - SHA-256 provenance hashes on every extraction operation
    - All anomaly flags use configurable, deterministic thresholds

Regulatory References:
    - EU 2023/1115 (EUDR) Article 4: Document authentication requirements
    - EU 2023/1115 (EUDR) Article 14: Five-year record retention
    - eIDAS Regulation (EU) No 910/2014: Digital document metadata standards
    - ISO 32000-2:2020 (PDF 2.0): PDF metadata specification
    - Exif 2.32 Standard: EXIF metadata for embedded images
    - ISO 16684-1:2019: XMP metadata specification

Performance Targets:
    - Single PDF metadata extraction: <50ms
    - Single EXIF extraction: <30ms
    - Single XMP extraction: <20ms
    - Metadata validation: <10ms
    - Batch extraction (100 documents): <2s

PRD Feature References:
    - PRD-AGENT-EUDR-012 Feature 5: Metadata Extraction and Validation
    - PRD-AGENT-EUDR-012 Feature 5.1: PDF Metadata Extraction
    - PRD-AGENT-EUDR-012 Feature 5.2: EXIF Metadata from Embedded Images
    - PRD-AGENT-EUDR-012 Feature 5.3: XMP Metadata Extraction
    - PRD-AGENT-EUDR-012 Feature 5.4: Creation Date Cross-Validation
    - PRD-AGENT-EUDR-012 Feature 5.5: Author/Producer Cross-Validation
    - PRD-AGENT-EUDR-012 Feature 5.6: Metadata Stripping Detection
    - PRD-AGENT-EUDR-012 Feature 5.7: Tool Mismatch Detection
    - PRD-AGENT-EUDR-012 Feature 5.8: Serial Number Extraction
    - PRD-AGENT-EUDR-012 Feature 5.9: GPS Coordinate Extraction
    - PRD-AGENT-EUDR-012 Feature 5.10: SHA-256 Provenance

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-012
Agent ID: GL-EUDR-DAV-012
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.document_authentication.config import (
    DocumentAuthenticationConfig,
    get_config,
)
from greenlang.agents.eudr.document_authentication.models import (
    DocumentType,
    MetadataField,
    MetadataRecord,
    MetadataResponse,
)
from greenlang.agents.eudr.document_authentication.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
)
from greenlang.agents.eudr.document_authentication.metrics import (
    observe_verification_duration,
    record_api_error,
    record_document_processed,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance.

    Args:
        data: Any JSON-serializable object.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id(prefix: str = "META") -> str:
    """Generate a prefixed UUID4 string identifier.

    Args:
        prefix: String prefix for the identifier.

    Returns:
        Prefixed UUID4 string.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Known PDF producers / creators for mismatch detection
# ---------------------------------------------------------------------------

#: Known legitimate PDF producers for EUDR document types.
KNOWN_PDF_PRODUCERS: Dict[str, List[str]] = {
    "coo": [
        "Adobe Acrobat", "Adobe PDF Library",
        "Microsoft Print to PDF", "OpenSSL",
        "iText", "wkhtmltopdf", "Prince",
    ],
    "pc": [
        "Adobe Acrobat", "iText", "IPPC ePhyto System",
        "National Plant Protection Organization",
        "wkhtmltopdf", "JasperReports",
    ],
    "bol": [
        "Adobe Acrobat", "CargoSmart", "INTTRA",
        "Bolero International", "essDOCS",
        "iText", "JasperReports",
    ],
    "fsc_cert": [
        "FSC Certificate Generator", "Adobe Acrobat",
        "iText", "Accreditation Services International",
    ],
    "rspo_cert": [
        "RSPO PalmTrace", "Adobe Acrobat", "iText",
        "UTZ Certification System",
    ],
    "iscc_cert": [
        "ISCC Certificate System", "Adobe Acrobat", "iText",
    ],
}

#: Known serial number formats per document type (regex patterns).
SERIAL_NUMBER_PATTERNS: Dict[str, List[str]] = {
    "fsc_cert": [
        r"FSC-C\d{6}",              # FSC CoC license code
        r"CU-COC-\d{6}",            # Control Union CoC
        r"SA-COC-\d{6}",            # Soil Association CoC
        r"BV-COC-\d{6}",            # Bureau Veritas CoC
        r"SGS-COC-\d{6}",           # SGS CoC
        r"TT-COC-\d{6}",           # TUV NORD CoC
        r"IC-COC-\d{6}",           # ICILA CoC
    ],
    "rspo_cert": [
        r"RSPO-\d{7}",              # RSPO membership number
        r"P&C-\d{4}-\d{6}",        # P&C certificate
        r"SCCS-\d{4}-\d{6}",       # SCCS certificate
    ],
    "iscc_cert": [
        r"ISCC-CERT-[A-Z]{2}\d{3}-\d{8}",  # ISCC cert format
        r"EU-ISCC-Cert-[A-Z]{2}\d{5}",      # EU variant
    ],
    "coo": [
        r"[A-Z]{2}-COO-\d{4}-\d{6}",       # Country-COO-Year-Seq
        r"COO/\d{4}/\d{6,8}",               # COO/Year/Sequence
    ],
    "pc": [
        r"[A-Z]{2,3}\d{2}[A-Z]\d{6,10}",   # Country + Year + Seq
        r"PC-\d{4}-\d{8}",                   # PC-Year-Sequence
        r"ePhyto-\d{10,14}",                # ePhyto electronic
    ],
    "bol": [
        r"[A-Z]{4}\d{10,12}",               # Carrier prefix + number
        r"BL-\d{4}-\d{8}",                  # BL-Year-Sequence
        r"MBOL\d{12}",                       # Master BOL
    ],
}

#: Expected required metadata fields per file format.
EXPECTED_METADATA_BY_FORMAT: Dict[str, List[str]] = {
    "pdf": ["title", "author", "creator", "producer", "creation_date"],
    "tiff": ["creation_date", "software"],
    "jpeg": ["creation_date", "camera_model"],
    "png": ["creation_date"],
    "docx": ["title", "author", "creation_date"],
}

#: MIME type to file format mapping.
MIME_TO_FORMAT: Dict[str, str] = {
    "application/pdf": "pdf",
    "image/tiff": "tiff",
    "image/jpeg": "jpeg",
    "image/png": "png",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "docx",
}

#: GPS DMS (degrees, minutes, seconds) regex pattern.
_GPS_DMS_PATTERN = re.compile(
    r"(\d{1,3})[^\d]+(\d{1,2})[^\d]+([\d.]+)[^\d]*([NSEW])"
)

#: Date format patterns commonly found in PDF metadata.
_PDF_DATE_PATTERNS: List[str] = [
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y%m%d%H%M%S",
    "D:%Y%m%d%H%M%S",
    "D:%Y%m%d%H%M%S%z",
]

#: Suspicious creator tools that indicate potential forgery.
SUSPICIOUS_CREATORS: List[str] = [
    "GIMP",
    "Photoshop",
    "Paint",
    "Inkscape",
    "CorelDRAW",
    "Preview",
    "ImageMagick",
    "PDFCreator",
]


# ---------------------------------------------------------------------------
# MetadataExtractorEngine
# ---------------------------------------------------------------------------


class MetadataExtractorEngine:
    """Engine for extracting and validating document metadata for EUDR authentication.

    Provides comprehensive metadata extraction from PDF, EXIF, and XMP
    sources with cross-validation against claimed document attributes.
    Detects metadata anomalies including stripping, tool mismatch,
    creation date inconsistency, and author mismatch.

    All operations are thread-safe via reentrant locking. All extractions
    use deterministic parsing with SHA-256 provenance hashing for
    zero-hallucination compliance.

    Attributes:
        _config: Document authentication configuration.
        _provenance: ProvenanceTracker for audit trail.
        _metadata_store: In-memory metadata record storage keyed by
            document_id.
        _extraction_index: Index mapping extraction ID to document ID.
        _lock: Reentrant lock for thread-safe access.

    Example:
        >>> engine = MetadataExtractorEngine()
        >>> result = engine.extract_metadata(
        ...     document_bytes=b"...",
        ...     filename="cert.pdf",
        ...     mime_type="application/pdf",
        ...     document_id="doc-001",
        ... )
        >>> assert result["success"] is True
    """

    def __init__(
        self,
        config: Optional[DocumentAuthenticationConfig] = None,
    ) -> None:
        """Initialize MetadataExtractorEngine.

        Args:
            config: Optional configuration override. If None, the
                singleton configuration from ``get_config()`` is used.
        """
        self._config: DocumentAuthenticationConfig = config or get_config()
        self._provenance: ProvenanceTracker = get_provenance_tracker()

        # In-memory storage
        self._metadata_store: Dict[str, Dict[str, Any]] = {}
        self._extraction_index: Dict[str, str] = {}

        # Thread safety
        self._lock: threading.RLock = threading.RLock()

        logger.info(
            "MetadataExtractorEngine initialized: "
            "creation_date_tolerance=%dd, require_author_match=%s, "
            "flag_empty_metadata=%s",
            self._config.creation_date_tolerance_days,
            self._config.require_author_match,
            self._config.flag_empty_metadata,
        )

    # ------------------------------------------------------------------
    # Public API: Extract metadata
    # ------------------------------------------------------------------

    def extract_metadata(
        self,
        document_bytes: bytes,
        filename: str,
        mime_type: str,
        document_id: Optional[str] = None,
        claimed_issuance_date: Optional[datetime] = None,
        claimed_author: Optional[str] = None,
        claimed_issuing_authority: Optional[str] = None,
        upload_date: Optional[datetime] = None,
        document_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract and validate document metadata for authentication.

        Performs a multi-stage extraction pipeline:
        1. Determine file format from MIME type
        2. Extract PDF metadata (Title, Author, Creator, Producer, dates)
        3. Extract EXIF metadata from embedded images (GPS, camera, date)
        4. Extract XMP metadata if present
        5. Extract serial/reference numbers
        6. Cross-validate against claimed attributes
        7. Detect anomalies (stripping, tool mismatch, date issues)
        8. Record provenance hash

        Args:
            document_bytes: Raw file content as bytes.
            filename: Original filename of the document.
            mime_type: MIME type of the document.
            document_id: Optional document identifier. Generated if None.
            claimed_issuance_date: Date the document claims to have been
                issued, for cross-validation with creation date.
            claimed_author: Claimed document author for cross-validation.
            claimed_issuing_authority: Claimed issuing authority for
                cross-validation with producer/creator metadata.
            upload_date: Date when the document was uploaded to the
                system, for creation date tolerance checking.
            document_type: EUDR document type (coo, fsc_cert, etc.)
                for type-specific validation.

        Returns:
            Dictionary with keys: success, extraction_id, document_id,
            metadata (MetadataRecord dict), anomalies, missing_fields,
            serial_numbers, gps_coordinates, validation_summary,
            processing_time_ms, provenance_hash.

        Raises:
            ValueError: If document_bytes is empty or filename is empty.
        """
        start_time = time.monotonic()

        if not document_bytes:
            raise ValueError("document_bytes must not be empty")
        if not filename:
            raise ValueError("filename must not be empty")

        doc_id = document_id or str(uuid.uuid4())
        extraction_id = _generate_id("META")

        logger.info(
            "Extracting metadata: document_id=%s, filename=%s, "
            "mime_type=%s, size=%d bytes",
            doc_id[:16], filename, mime_type, len(document_bytes),
        )

        try:
            # Step 1: Determine file format
            file_format = self._determine_format(mime_type, filename)

            # Step 2: Extract PDF metadata
            pdf_metadata = self._extract_pdf_metadata(
                document_bytes, file_format,
            )

            # Step 3: Extract EXIF metadata
            exif_metadata = self._extract_exif_metadata(
                document_bytes, file_format,
            )

            # Step 4: Extract XMP metadata
            xmp_metadata = self._extract_xmp_metadata(
                document_bytes, file_format,
            )

            # Step 5: Merge all metadata sources
            merged = self._merge_metadata(
                pdf_metadata, exif_metadata, xmp_metadata,
            )

            # Step 6: Extract serial/reference numbers
            serial_numbers = self._extract_serial_numbers(
                document_bytes, document_type, merged,
            )

            # Step 7: Extract dates from content
            extracted_dates = self._extract_dates(merged)

            # Step 8: Extract GPS coordinates
            gps_coords = self._extract_gps_coordinates(
                exif_metadata, xmp_metadata, merged,
            )

            # Step 9: Detect anomalies
            anomalies: List[str] = []

            # 9a: Check for metadata stripping
            stripping_result = self._detect_stripping(
                merged, file_format,
            )
            anomalies.extend(stripping_result)

            # 9b: Check for tool mismatch
            tool_mismatch = self._detect_tool_mismatch(
                merged, document_type,
            )
            anomalies.extend(tool_mismatch)

            # 9c: Validate creation date
            date_anomalies = self._validate_creation_date(
                merged, claimed_issuance_date, upload_date,
            )
            anomalies.extend(date_anomalies)

            # 9d: Validate author
            author_anomalies = self._validate_author(
                merged, claimed_author, claimed_issuing_authority,
            )
            anomalies.extend(author_anomalies)

            # Step 10: Check missing required fields
            missing_fields = self._check_missing_fields(
                merged, file_format,
            )

            # Step 11: Determine creation date anomaly flag
            creation_date_anomaly = len(date_anomalies) > 0

            # Step 12: Determine author match flag
            author_match = self._compute_author_match(
                merged, claimed_author,
            )

            # Step 13: Build metadata record
            now = _utcnow()
            creation_date = self._parse_date_field(
                merged.get("creation_date"),
            )
            modification_date = self._parse_date_field(
                merged.get("modification_date"),
            )

            metadata_dict: Dict[str, Any] = {
                "document_id": doc_id,
                "title": merged.get("title"),
                "author": merged.get("author"),
                "creator": merged.get("creator"),
                "producer": merged.get("producer"),
                "creation_date": (
                    creation_date.isoformat() if creation_date else None
                ),
                "modification_date": (
                    modification_date.isoformat()
                    if modification_date else None
                ),
                "keywords": merged.get("keywords", []),
                "gps_lat": gps_coords.get("latitude"),
                "gps_lon": gps_coords.get("longitude"),
                "page_count": merged.get("page_count", 0),
                "file_format": file_format,
                "raw_metadata": merged,
                "anomalies": anomalies,
                "missing_fields": missing_fields,
                "creation_date_anomaly": creation_date_anomaly,
                "author_match": author_match,
                "extracted_at": now.isoformat(),
            }

            # Step 14: Build validation summary
            validation_summary = self._build_validation_summary(
                anomalies, missing_fields, creation_date_anomaly,
                author_match, serial_numbers, gps_coords,
            )

            # Step 15: Compute provenance hash
            provenance_data = {
                "extraction_id": extraction_id,
                "document_id": doc_id,
                "metadata": metadata_dict,
                "serial_numbers": serial_numbers,
                "gps_coordinates": gps_coords,
                "validation_summary": validation_summary,
                "module_version": _MODULE_VERSION,
            }
            provenance_hash = _compute_hash(provenance_data)
            metadata_dict["provenance_hash"] = provenance_hash

            # Step 16: Record provenance
            if self._config.enable_provenance:
                self._provenance.record(
                    entity_type="metadata",
                    action="extract_metadata",
                    entity_id=doc_id,
                    data=provenance_data,
                    metadata={
                        "extraction_id": extraction_id,
                        "document_id": doc_id,
                        "file_format": file_format,
                        "anomaly_count": len(anomalies),
                        "missing_field_count": len(missing_fields),
                    },
                )

            # Step 17: Store result
            elapsed_ms = (time.monotonic() - start_time) * 1000

            result: Dict[str, Any] = {
                "success": True,
                "extraction_id": extraction_id,
                "document_id": doc_id,
                "metadata": metadata_dict,
                "anomalies": anomalies,
                "missing_fields": missing_fields,
                "serial_numbers": serial_numbers,
                "gps_coordinates": gps_coords,
                "extracted_dates": extracted_dates,
                "validation_summary": validation_summary,
                "processing_time_ms": round(elapsed_ms, 2),
                "provenance_hash": provenance_hash,
            }

            with self._lock:
                self._metadata_store[doc_id] = result
                self._extraction_index[extraction_id] = doc_id

            # Step 18: Record metrics
            if self._config.enable_metrics:
                observe_verification_duration(elapsed_ms / 1000)
                record_document_processed(document_type or "unknown")

            logger.info(
                "Metadata extraction completed: document_id=%s, "
                "extraction_id=%s, anomalies=%d, missing=%d, "
                "serials=%d, gps=%s, elapsed=%.1fms",
                doc_id[:16], extraction_id[:16],
                len(anomalies), len(missing_fields),
                len(serial_numbers),
                "yes" if gps_coords.get("latitude") else "no",
                elapsed_ms,
            )

            return result

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Metadata extraction failed: document_id=%s, "
                "error=%s, elapsed=%.1fms",
                doc_id[:16], str(exc), elapsed_ms,
                exc_info=True,
            )
            if self._config.enable_metrics:
                record_api_error("extract_metadata")
            return {
                "success": False,
                "extraction_id": extraction_id,
                "document_id": doc_id,
                "metadata": None,
                "anomalies": [],
                "missing_fields": [],
                "serial_numbers": [],
                "gps_coordinates": {},
                "extracted_dates": [],
                "validation_summary": {"status": "error", "error": str(exc)},
                "processing_time_ms": round(elapsed_ms, 2),
                "provenance_hash": None,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Public API: Validate metadata
    # ------------------------------------------------------------------

    def validate_metadata(
        self,
        metadata: Dict[str, Any],
        claimed_attributes: Dict[str, Any],
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate extracted metadata against claimed document attributes.

        Performs cross-validation of metadata fields against the set of
        attributes the submitting operator claims for the document.

        Args:
            metadata: Extracted metadata dictionary (from extract_metadata).
            claimed_attributes: Dictionary of claimed attributes, supporting
                keys: issuance_date, author, issuing_authority, serial_number,
                commodity, origin_country, quantity, document_type.
            document_id: Optional document identifier for provenance.

        Returns:
            Dictionary with keys: valid, document_id, validation_errors,
            validation_warnings, cross_validation_results,
            processing_time_ms, provenance_hash.
        """
        start_time = time.monotonic()
        doc_id = document_id or metadata.get("document_id", str(uuid.uuid4()))
        validation_id = _generate_id("MVAL")

        logger.info(
            "Validating metadata: document_id=%s, claimed_keys=%s",
            doc_id[:16], list(claimed_attributes.keys()),
        )

        errors: List[str] = []
        warnings: List[str] = []
        cross_results: Dict[str, Dict[str, Any]] = {}

        try:
            # Validate issuance date
            if "issuance_date" in claimed_attributes:
                date_result = self._cross_validate_date(
                    metadata, claimed_attributes["issuance_date"],
                )
                cross_results["issuance_date"] = date_result
                if not date_result.get("match"):
                    if date_result.get("severity") == "error":
                        errors.append(date_result["message"])
                    else:
                        warnings.append(date_result["message"])

            # Validate author
            if "author" in claimed_attributes:
                author_result = self._cross_validate_author(
                    metadata, claimed_attributes["author"],
                )
                cross_results["author"] = author_result
                if not author_result.get("match"):
                    if self._config.require_author_match:
                        errors.append(author_result["message"])
                    else:
                        warnings.append(author_result["message"])

            # Validate issuing authority
            if "issuing_authority" in claimed_attributes:
                authority_result = self._cross_validate_authority(
                    metadata, claimed_attributes["issuing_authority"],
                )
                cross_results["issuing_authority"] = authority_result
                if not authority_result.get("match"):
                    warnings.append(authority_result["message"])

            # Validate serial number
            if "serial_number" in claimed_attributes:
                serial_result = self._cross_validate_serial(
                    metadata, claimed_attributes["serial_number"],
                    claimed_attributes.get("document_type"),
                )
                cross_results["serial_number"] = serial_result
                if not serial_result.get("match"):
                    errors.append(serial_result["message"])

            # Check empty metadata flag
            if self._config.flag_empty_metadata:
                raw = metadata.get("raw_metadata", {})
                if not raw or len(raw) < 2:
                    warnings.append(
                        "Document has minimal or no metadata; "
                        "possible metadata stripping detected"
                    )

            is_valid = len(errors) == 0
            elapsed_ms = (time.monotonic() - start_time) * 1000

            # Provenance
            provenance_data = {
                "validation_id": validation_id,
                "document_id": doc_id,
                "is_valid": is_valid,
                "errors": errors,
                "warnings": warnings,
                "cross_results": cross_results,
                "module_version": _MODULE_VERSION,
            }
            provenance_hash = _compute_hash(provenance_data)

            if self._config.enable_provenance:
                self._provenance.record(
                    entity_type="metadata",
                    action="extract_metadata",
                    entity_id=doc_id,
                    data=provenance_data,
                    metadata={
                        "validation_id": validation_id,
                        "document_id": doc_id,
                        "is_valid": is_valid,
                        "error_count": len(errors),
                        "warning_count": len(warnings),
                    },
                )

            logger.info(
                "Metadata validation completed: document_id=%s, "
                "valid=%s, errors=%d, warnings=%d, elapsed=%.1fms",
                doc_id[:16], is_valid, len(errors), len(warnings),
                elapsed_ms,
            )

            return {
                "valid": is_valid,
                "validation_id": validation_id,
                "document_id": doc_id,
                "validation_errors": errors,
                "validation_warnings": warnings,
                "cross_validation_results": cross_results,
                "processing_time_ms": round(elapsed_ms, 2),
                "provenance_hash": provenance_hash,
            }

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Metadata validation failed: document_id=%s, error=%s",
                doc_id[:16], str(exc), exc_info=True,
            )
            if self._config.enable_metrics:
                record_api_error("extract_metadata")
            return {
                "valid": False,
                "validation_id": validation_id,
                "document_id": doc_id,
                "validation_errors": [f"Validation error: {str(exc)}"],
                "validation_warnings": [],
                "cross_validation_results": {},
                "processing_time_ms": round(elapsed_ms, 2),
                "provenance_hash": None,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Public API: Batch extraction
    # ------------------------------------------------------------------

    def batch_extract_metadata(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract metadata from multiple documents.

        Args:
            documents: List of dictionaries, each containing at minimum:
                document_bytes (bytes), filename (str), mime_type (str).
                Optional keys: document_id, claimed_issuance_date,
                claimed_author, claimed_issuing_authority, upload_date,
                document_type.

        Returns:
            List of extraction result dictionaries (same format as
            extract_metadata output).

        Raises:
            ValueError: If documents list is empty or exceeds batch limit.
        """
        if not documents:
            raise ValueError("documents list must not be empty")

        max_size = self._config.batch_max_size
        if len(documents) > max_size:
            raise ValueError(
                f"Batch size {len(documents)} exceeds maximum {max_size}"
            )

        logger.info(
            "Batch metadata extraction: %d documents", len(documents),
        )

        results: List[Dict[str, Any]] = []
        for doc in documents:
            result = self.extract_metadata(
                document_bytes=doc.get("document_bytes", b""),
                filename=doc.get("filename", "unknown"),
                mime_type=doc.get("mime_type", "application/octet-stream"),
                document_id=doc.get("document_id"),
                claimed_issuance_date=doc.get("claimed_issuance_date"),
                claimed_author=doc.get("claimed_author"),
                claimed_issuing_authority=doc.get(
                    "claimed_issuing_authority",
                ),
                upload_date=doc.get("upload_date"),
                document_type=doc.get("document_type"),
            )
            results.append(result)

        succeeded = sum(1 for r in results if r.get("success"))
        logger.info(
            "Batch metadata extraction completed: %d/%d succeeded",
            succeeded, len(documents),
        )
        return results

    # ------------------------------------------------------------------
    # Public API: Get stored metadata
    # ------------------------------------------------------------------

    def get_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored metadata for a document.

        Args:
            document_id: Document identifier.

        Returns:
            Metadata extraction result or None if not found.
        """
        with self._lock:
            return self._metadata_store.get(document_id)

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Return extraction statistics.

        Returns:
            Dictionary with total_extractions, anomaly_distribution,
            format_distribution, and missing_field_distribution.
        """
        with self._lock:
            records = list(self._metadata_store.values())

        total = len(records)
        anomaly_counts: Dict[str, int] = {}
        format_counts: Dict[str, int] = {}
        total_anomalies = 0
        total_missing = 0

        for record in records:
            if not record.get("success"):
                continue
            meta = record.get("metadata", {})
            fmt = meta.get("file_format", "unknown")
            format_counts[fmt] = format_counts.get(fmt, 0) + 1

            for anomaly in record.get("anomalies", []):
                key = anomaly[:50]
                anomaly_counts[key] = anomaly_counts.get(key, 0) + 1
                total_anomalies += 1

            total_missing += len(record.get("missing_fields", []))

        return {
            "total_extractions": total,
            "total_anomalies": total_anomalies,
            "total_missing_fields": total_missing,
            "anomaly_distribution": anomaly_counts,
            "format_distribution": format_counts,
        }

    # ------------------------------------------------------------------
    # PDF metadata extraction
    # ------------------------------------------------------------------

    def extract_pdf_metadata(
        self,
        document_bytes: bytes,
    ) -> Dict[str, Any]:
        """Extract metadata fields from a PDF document.

        Parses the PDF info dictionary and document catalog to extract
        standard metadata fields: Title, Author, Subject, Keywords,
        Creator, Producer, CreationDate, ModDate.

        This method uses deterministic byte-level parsing of PDF
        structures without relying on external libraries. In production,
        it would integrate with a PDF library; this implementation
        uses pattern-based extraction for the deterministic path.

        Args:
            document_bytes: Raw PDF file content.

        Returns:
            Dictionary of extracted PDF metadata fields.
        """
        metadata: Dict[str, Any] = {}

        if not document_bytes or len(document_bytes) < 5:
            return metadata

        # Check PDF magic bytes
        content = document_bytes[:min(len(document_bytes), 65536)]
        try:
            text = content.decode("latin-1", errors="replace")
        except Exception:
            return metadata

        is_pdf = text[:5] == "%PDF-"
        if not is_pdf:
            return metadata

        # Extract PDF version
        version_match = re.search(r"%PDF-(\d+\.\d+)", text)
        if version_match:
            metadata["pdf_version"] = version_match.group(1)

        # Extract info dictionary fields via regex
        # Title
        title_match = re.search(
            r"/Title\s*\(([^)]*)\)", text,
        )
        if title_match:
            metadata["title"] = title_match.group(1).strip()

        # Author
        author_match = re.search(
            r"/Author\s*\(([^)]*)\)", text,
        )
        if author_match:
            metadata["author"] = author_match.group(1).strip()

        # Creator
        creator_match = re.search(
            r"/Creator\s*\(([^)]*)\)", text,
        )
        if creator_match:
            metadata["creator"] = creator_match.group(1).strip()

        # Producer
        producer_match = re.search(
            r"/Producer\s*\(([^)]*)\)", text,
        )
        if producer_match:
            metadata["producer"] = producer_match.group(1).strip()

        # Subject
        subject_match = re.search(
            r"/Subject\s*\(([^)]*)\)", text,
        )
        if subject_match:
            metadata["subject"] = subject_match.group(1).strip()

        # Keywords
        keywords_match = re.search(
            r"/Keywords\s*\(([^)]*)\)", text,
        )
        if keywords_match:
            raw_kw = keywords_match.group(1).strip()
            metadata["keywords"] = [
                k.strip() for k in re.split(r"[,;]", raw_kw) if k.strip()
            ]

        # CreationDate
        creation_match = re.search(
            r"/CreationDate\s*\(([^)]*)\)", text,
        )
        if creation_match:
            metadata["creation_date"] = creation_match.group(1).strip()

        # ModDate
        mod_match = re.search(
            r"/ModDate\s*\(([^)]*)\)", text,
        )
        if mod_match:
            metadata["modification_date"] = mod_match.group(1).strip()

        # Page count (approximate by counting /Type /Page entries)
        page_count = len(re.findall(r"/Type\s*/Page(?!\s*s)", text))
        metadata["page_count"] = max(page_count, 1) if is_pdf else 0

        logger.debug(
            "PDF metadata extracted: fields=%d, pages=%d",
            len(metadata), metadata.get("page_count", 0),
        )
        return metadata

    # ------------------------------------------------------------------
    # EXIF metadata extraction
    # ------------------------------------------------------------------

    def extract_exif_metadata(
        self,
        document_bytes: bytes,
    ) -> Dict[str, Any]:
        """Extract EXIF metadata from embedded images in documents.

        Parses EXIF headers from JPEG images or embedded image streams
        in PDF documents to extract GPS coordinates, camera model,
        capture date, and orientation data.

        Uses deterministic byte-level parsing of EXIF structures.

        Args:
            document_bytes: Raw file content.

        Returns:
            Dictionary of extracted EXIF fields including gps_lat,
            gps_lon, camera_model, capture_date, orientation.
        """
        exif: Dict[str, Any] = {}

        if not document_bytes or len(document_bytes) < 12:
            return exif

        # Check for JPEG SOI marker (FFD8FF)
        is_jpeg = (
            document_bytes[0:2] == b"\xff\xd8"
        )

        # Check for EXIF header within first 64KB
        content = document_bytes[:min(len(document_bytes), 65536)]

        # Look for Exif header marker
        exif_start = content.find(b"Exif\x00\x00")

        if exif_start < 0 and not is_jpeg:
            # For PDFs, look for embedded JPEG streams
            jpeg_start = content.find(b"\xff\xd8\xff")
            if jpeg_start >= 0:
                embedded_jpeg = content[jpeg_start:]
                exif_start = embedded_jpeg.find(b"Exif\x00\x00")
                if exif_start >= 0:
                    content = embedded_jpeg

        if exif_start < 0:
            return exif

        # Parse EXIF text representation for key fields
        try:
            text_content = content.decode("latin-1", errors="replace")
        except Exception:
            return exif

        # GPS Latitude (look for GPS IFD patterns)
        gps_data = self._parse_exif_gps(content, text_content)
        if gps_data:
            exif.update(gps_data)

        # Camera model
        model_match = re.search(
            r"(?:Model|Camera)\x00+([^\x00]{3,50})", text_content,
        )
        if model_match:
            model_str = model_match.group(1).strip()
            if model_str.isprintable():
                exif["camera_model"] = model_str

        # Software
        software_match = re.search(
            r"Software\x00+([^\x00]{3,50})", text_content,
        )
        if software_match:
            sw_str = software_match.group(1).strip()
            if sw_str.isprintable():
                exif["software"] = sw_str

        # DateTime
        datetime_match = re.search(
            r"DateTime\x00+(\d{4}:\d{2}:\d{2}\s+\d{2}:\d{2}:\d{2})",
            text_content,
        )
        if datetime_match:
            exif["capture_date"] = datetime_match.group(1).strip()

        # Orientation
        orientation_match = re.search(
            r"Orientation\x00+(\d)", text_content,
        )
        if orientation_match:
            exif["orientation"] = int(orientation_match.group(1))

        # Image dimensions
        width_match = re.search(
            r"ImageWidth\x00+(\d+)", text_content,
        )
        if width_match:
            exif["image_width"] = int(width_match.group(1))

        height_match = re.search(
            r"ImageLength\x00+(\d+)", text_content,
        )
        if height_match:
            exif["image_height"] = int(height_match.group(1))

        logger.debug(
            "EXIF metadata extracted: fields=%d, gps=%s",
            len(exif), "yes" if "gps_lat" in exif else "no",
        )
        return exif

    # ------------------------------------------------------------------
    # XMP metadata extraction
    # ------------------------------------------------------------------

    def extract_xmp_metadata(
        self,
        document_bytes: bytes,
    ) -> Dict[str, Any]:
        """Extract XMP metadata from document bytes.

        Parses XMP (Extensible Metadata Platform) packets embedded in
        PDF, JPEG, and TIFF files. XMP data is stored as XML within
        the file.

        Args:
            document_bytes: Raw file content.

        Returns:
            Dictionary of extracted XMP fields.
        """
        xmp: Dict[str, Any] = {}

        if not document_bytes or len(document_bytes) < 20:
            return xmp

        # XMP packets are wrapped in <?xpacket begin=...?> ... <?xpacket end=...?>
        try:
            content = document_bytes.decode("latin-1", errors="replace")
        except Exception:
            return xmp

        # Find XMP packet boundaries
        xmp_begin = content.find("<?xpacket begin=")
        xmp_end = content.find("<?xpacket end=")

        if xmp_begin < 0 or xmp_end < 0:
            # Try looking for x:xmpmeta directly
            xmp_begin = content.find("<x:xmpmeta")
            xmp_end = content.find("</x:xmpmeta>")
            if xmp_begin < 0 or xmp_end < 0:
                return xmp

        xmp_data = content[xmp_begin:xmp_end + 100]

        # Extract dc:title
        title_match = re.search(
            r"<dc:title[^>]*>.*?<rdf:li[^>]*>([^<]+)</rdf:li>",
            xmp_data, re.DOTALL,
        )
        if title_match:
            xmp["title"] = title_match.group(1).strip()

        # Extract dc:creator
        creator_match = re.search(
            r"<dc:creator[^>]*>.*?<rdf:li[^>]*>([^<]+)</rdf:li>",
            xmp_data, re.DOTALL,
        )
        if creator_match:
            xmp["creator"] = creator_match.group(1).strip()

        # Extract dc:description
        desc_match = re.search(
            r"<dc:description[^>]*>.*?<rdf:li[^>]*>([^<]+)</rdf:li>",
            xmp_data, re.DOTALL,
        )
        if desc_match:
            xmp["description"] = desc_match.group(1).strip()

        # Extract xmp:CreateDate
        create_date_match = re.search(
            r"<xmp:CreateDate>([^<]+)</xmp:CreateDate>", xmp_data,
        )
        if create_date_match:
            xmp["creation_date"] = create_date_match.group(1).strip()

        # Extract xmp:ModifyDate
        modify_date_match = re.search(
            r"<xmp:ModifyDate>([^<]+)</xmp:ModifyDate>", xmp_data,
        )
        if modify_date_match:
            xmp["modification_date"] = modify_date_match.group(1).strip()

        # Extract xmp:CreatorTool
        tool_match = re.search(
            r"<xmp:CreatorTool>([^<]+)</xmp:CreatorTool>", xmp_data,
        )
        if tool_match:
            xmp["creator_tool"] = tool_match.group(1).strip()

        # Extract pdf:Producer
        producer_match = re.search(
            r"<pdf:Producer>([^<]+)</pdf:Producer>", xmp_data,
        )
        if producer_match:
            xmp["producer"] = producer_match.group(1).strip()

        # Extract dc:subject (keywords)
        keywords: List[str] = []
        keyword_matches = re.findall(
            r"<dc:subject[^>]*>.*?</dc:subject>", xmp_data, re.DOTALL,
        )
        for kw_block in keyword_matches:
            kw_items = re.findall(r"<rdf:li[^>]*>([^<]+)</rdf:li>", kw_block)
            keywords.extend(k.strip() for k in kw_items if k.strip())
        if keywords:
            xmp["keywords"] = keywords

        # Extract GPS from XMP (exif:GPSLatitude, exif:GPSLongitude)
        lat_match = re.search(
            r"<exif:GPSLatitude>([^<]+)</exif:GPSLatitude>", xmp_data,
        )
        lon_match = re.search(
            r"<exif:GPSLongitude>([^<]+)</exif:GPSLongitude>", xmp_data,
        )
        if lat_match and lon_match:
            lat_val = self._parse_xmp_gps_coord(lat_match.group(1).strip())
            lon_val = self._parse_xmp_gps_coord(lon_match.group(1).strip())
            if lat_val is not None:
                xmp["gps_lat"] = lat_val
            if lon_val is not None:
                xmp["gps_lon"] = lon_val

        logger.debug(
            "XMP metadata extracted: fields=%d", len(xmp),
        )
        return xmp

    # ------------------------------------------------------------------
    # Internal: Merge metadata sources
    # ------------------------------------------------------------------

    def _merge_metadata(
        self,
        pdf_meta: Dict[str, Any],
        exif_meta: Dict[str, Any],
        xmp_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge metadata from PDF, EXIF, and XMP sources.

        Priority order: XMP > PDF > EXIF for overlapping fields,
        as XMP is typically the most recently written metadata.

        Args:
            pdf_meta: PDF info dictionary metadata.
            exif_meta: EXIF metadata from embedded images.
            xmp_meta: XMP metadata packet.

        Returns:
            Merged metadata dictionary.
        """
        merged: Dict[str, Any] = {}

        # Start with EXIF (lowest priority)
        merged.update(exif_meta)

        # Overlay PDF metadata
        for key, value in pdf_meta.items():
            if value is not None and value != "":
                merged[key] = value

        # Overlay XMP metadata (highest priority)
        for key, value in xmp_meta.items():
            if value is not None and value != "":
                merged[key] = value

        # Track source of each field
        sources: Dict[str, str] = {}
        for key in merged:
            if key in xmp_meta and xmp_meta[key]:
                sources[key] = "xmp"
            elif key in pdf_meta and pdf_meta[key]:
                sources[key] = "pdf"
            elif key in exif_meta and exif_meta[key]:
                sources[key] = "exif"
        merged["_metadata_sources"] = sources

        return merged

    # ------------------------------------------------------------------
    # Internal: Determine file format
    # ------------------------------------------------------------------

    def _determine_format(
        self,
        mime_type: str,
        filename: str,
    ) -> str:
        """Determine the file format from MIME type and filename extension.

        Args:
            mime_type: MIME type string.
            filename: Original filename.

        Returns:
            Normalized file format string (pdf, jpeg, tiff, png, docx).
        """
        # Try MIME type first
        fmt = MIME_TO_FORMAT.get(mime_type.lower().strip())
        if fmt:
            return fmt

        # Fallback to filename extension
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        ext_map = {
            "pdf": "pdf",
            "jpg": "jpeg",
            "jpeg": "jpeg",
            "tif": "tiff",
            "tiff": "tiff",
            "png": "png",
            "docx": "docx",
            "doc": "docx",
        }
        return ext_map.get(ext, "unknown")

    # ------------------------------------------------------------------
    # Internal: Extract PDF metadata (wrapper for public method)
    # ------------------------------------------------------------------

    def _extract_pdf_metadata(
        self,
        document_bytes: bytes,
        file_format: str,
    ) -> Dict[str, Any]:
        """Extract PDF metadata if the document is a PDF.

        Args:
            document_bytes: Raw file content.
            file_format: Detected file format.

        Returns:
            PDF metadata dictionary (empty if not a PDF).
        """
        if file_format != "pdf":
            return {}
        return self.extract_pdf_metadata(document_bytes)

    # ------------------------------------------------------------------
    # Internal: Extract EXIF metadata (wrapper)
    # ------------------------------------------------------------------

    def _extract_exif_metadata(
        self,
        document_bytes: bytes,
        file_format: str,
    ) -> Dict[str, Any]:
        """Extract EXIF metadata if applicable.

        Args:
            document_bytes: Raw file content.
            file_format: Detected file format.

        Returns:
            EXIF metadata dictionary.
        """
        if file_format not in ("jpeg", "tiff", "pdf"):
            return {}
        return self.extract_exif_metadata(document_bytes)

    # ------------------------------------------------------------------
    # Internal: Extract XMP metadata (wrapper)
    # ------------------------------------------------------------------

    def _extract_xmp_metadata(
        self,
        document_bytes: bytes,
        file_format: str,
    ) -> Dict[str, Any]:
        """Extract XMP metadata if applicable.

        Args:
            document_bytes: Raw file content.
            file_format: Detected file format.

        Returns:
            XMP metadata dictionary.
        """
        if file_format not in ("pdf", "jpeg", "tiff", "png"):
            return {}
        return self.extract_xmp_metadata(document_bytes)

    # ------------------------------------------------------------------
    # Internal: Validate creation date
    # ------------------------------------------------------------------

    def _validate_creation_date(
        self,
        metadata: Dict[str, Any],
        claimed_issuance_date: Optional[datetime],
        upload_date: Optional[datetime],
    ) -> List[str]:
        """Cross-validate creation date against claimed issuance date.

        Flags anomalies when:
        - Creation date is AFTER the claimed issuance date (impossible)
        - Creation date differs from upload date by more than the
          configured tolerance
        - Modification date is before creation date

        Args:
            metadata: Merged metadata dictionary.
            claimed_issuance_date: Date the document claims issuance.
            upload_date: Date the document was uploaded.

        Returns:
            List of anomaly description strings.
        """
        anomalies: List[str] = []

        creation_date = self._parse_date_field(
            metadata.get("creation_date"),
        )
        modification_date = self._parse_date_field(
            metadata.get("modification_date"),
        )

        if creation_date and claimed_issuance_date:
            # Flag if creation date is AFTER claimed issuance date
            if creation_date > claimed_issuance_date:
                diff_days = (creation_date - claimed_issuance_date).days
                anomalies.append(
                    f"Creation date ({creation_date.date()}) is "
                    f"{diff_days} days AFTER claimed issuance date "
                    f"({claimed_issuance_date.date()}); document may "
                    f"have been created after the claimed issuance"
                )

            # Flag if creation date is too far before issuance
            tolerance = timedelta(
                days=self._config.creation_date_tolerance_days,
            )
            if claimed_issuance_date - creation_date > tolerance:
                diff_days = (claimed_issuance_date - creation_date).days
                anomalies.append(
                    f"Creation date ({creation_date.date()}) is "
                    f"{diff_days} days before claimed issuance date "
                    f"({claimed_issuance_date.date()}); exceeds "
                    f"{self._config.creation_date_tolerance_days}-day "
                    f"tolerance"
                )

        if creation_date and upload_date:
            tolerance = timedelta(
                days=self._config.creation_date_tolerance_days,
            )
            if upload_date - creation_date > tolerance:
                diff_days = (upload_date - creation_date).days
                anomalies.append(
                    f"Creation date ({creation_date.date()}) is "
                    f"{diff_days} days before upload date "
                    f"({upload_date.date()}); exceeds "
                    f"{self._config.creation_date_tolerance_days}-day "
                    f"tolerance"
                )

        # Check modification date vs creation date
        if creation_date and modification_date:
            if modification_date < creation_date:
                anomalies.append(
                    f"Modification date ({modification_date.date()}) "
                    f"is before creation date ({creation_date.date()}); "
                    f"metadata inconsistency"
                )

        return anomalies

    # ------------------------------------------------------------------
    # Internal: Validate author
    # ------------------------------------------------------------------

    def _validate_author(
        self,
        metadata: Dict[str, Any],
        claimed_author: Optional[str],
        claimed_issuing_authority: Optional[str],
    ) -> List[str]:
        """Cross-validate author/producer against claimed identity.

        Args:
            metadata: Merged metadata dictionary.
            claimed_author: Claimed document author.
            claimed_issuing_authority: Claimed issuing authority.

        Returns:
            List of anomaly description strings.
        """
        anomalies: List[str] = []

        doc_author = metadata.get("author", "")
        doc_producer = metadata.get("producer", "")
        doc_creator = metadata.get("creator", "")

        if claimed_author and doc_author:
            if not self._fuzzy_match(claimed_author, doc_author):
                anomalies.append(
                    f"Document author '{doc_author}' does not match "
                    f"claimed author '{claimed_author}'"
                )

        if claimed_issuing_authority:
            # Check if authority name appears in author, producer, or creator
            authority_lower = claimed_issuing_authority.lower()
            found_in_metadata = False

            for field_val in [doc_author, doc_producer, doc_creator]:
                if field_val and authority_lower in field_val.lower():
                    found_in_metadata = True
                    break

            if not found_in_metadata and doc_author:
                anomalies.append(
                    f"Claimed issuing authority "
                    f"'{claimed_issuing_authority}' not found in "
                    f"document metadata (author='{doc_author}', "
                    f"producer='{doc_producer}', "
                    f"creator='{doc_creator}')"
                )

        return anomalies

    # ------------------------------------------------------------------
    # Internal: Detect metadata stripping
    # ------------------------------------------------------------------

    def _detect_stripping(
        self,
        metadata: Dict[str, Any],
        file_format: str,
    ) -> List[str]:
        """Detect suspicious metadata stripping.

        A document with suspiciously empty metadata fields is flagged,
        as legitimate documents typically have creation tools, dates,
        and author information populated.

        Args:
            metadata: Merged metadata dictionary.
            file_format: Detected file format.

        Returns:
            List of anomaly description strings.
        """
        anomalies: List[str] = []

        if not self._config.flag_empty_metadata:
            return anomalies

        expected_fields = EXPECTED_METADATA_BY_FORMAT.get(
            file_format, ["creation_date"],
        )

        # Count populated fields (excluding internal keys)
        populated = 0
        for key, value in metadata.items():
            if key.startswith("_"):
                continue
            if value is not None and value != "" and value != []:
                populated += 1

        # Flag if fewer than 2 metadata fields are populated
        if populated < 2 and file_format == "pdf":
            anomalies.append(
                f"Suspiciously sparse metadata: only {populated} "
                f"field(s) populated in {file_format.upper()} document; "
                f"possible metadata stripping"
            )

        # Check specific expected fields
        missing_expected = []
        for field in expected_fields:
            val = metadata.get(field)
            if val is None or val == "":
                missing_expected.append(field)

        if len(missing_expected) == len(expected_fields) and expected_fields:
            anomalies.append(
                f"All expected metadata fields missing for "
                f"{file_format.upper()}: {missing_expected}; "
                f"possible metadata stripping"
            )

        return anomalies

    # ------------------------------------------------------------------
    # Internal: Detect tool mismatch
    # ------------------------------------------------------------------

    def _detect_tool_mismatch(
        self,
        metadata: Dict[str, Any],
        document_type: Optional[str],
    ) -> List[str]:
        """Detect creation tool mismatches for document type.

        Flags when the PDF creator or producer does not match the
        expected tools for the document type, which may indicate
        the document was forged using an image editor.

        Args:
            metadata: Merged metadata dictionary.
            document_type: EUDR document type identifier.

        Returns:
            List of anomaly description strings.
        """
        anomalies: List[str] = []

        creator = metadata.get("creator", "")
        producer = metadata.get("producer", "")
        creator_tool = metadata.get("creator_tool", "")

        # Check for known suspicious creators
        for tool in SUSPICIOUS_CREATORS:
            tool_lower = tool.lower()
            for field_name, field_val in [
                ("creator", creator),
                ("producer", producer),
                ("creator_tool", creator_tool),
            ]:
                if field_val and tool_lower in field_val.lower():
                    anomalies.append(
                        f"Document {field_name} '{field_val}' contains "
                        f"image editing tool '{tool}'; possible forgery "
                        f"using graphic manipulation software"
                    )

        # Check against known producers for document type
        if document_type and document_type in KNOWN_PDF_PRODUCERS:
            known = KNOWN_PDF_PRODUCERS[document_type]
            if producer:
                matched = any(
                    k.lower() in producer.lower() for k in known
                )
                if not matched:
                    anomalies.append(
                        f"PDF producer '{producer}' is not in the "
                        f"expected producer list for document type "
                        f"'{document_type}': {known}"
                    )

        return anomalies

    # ------------------------------------------------------------------
    # Internal: Extract serial numbers
    # ------------------------------------------------------------------

    def _extract_serial_numbers(
        self,
        document_bytes: bytes,
        document_type: Optional[str],
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Extract serial/reference/control numbers from document content.

        Uses regex patterns specific to each document type to find
        certificate numbers, reference numbers, and control numbers.

        Args:
            document_bytes: Raw file content.
            document_type: EUDR document type for pattern selection.
            metadata: Extracted metadata dictionary.

        Returns:
            List of dictionaries with keys: type, value, pattern,
            position, validated.
        """
        serials: List[Dict[str, Any]] = []

        # Try to extract text from the document bytes
        try:
            text = document_bytes[:min(len(document_bytes), 131072)].decode(
                "latin-1", errors="replace",
            )
        except Exception:
            return serials

        # Apply document-type-specific patterns
        if document_type and document_type in SERIAL_NUMBER_PATTERNS:
            patterns = SERIAL_NUMBER_PATTERNS[document_type]
            for pattern_str in patterns:
                try:
                    matches = re.finditer(pattern_str, text)
                    for match in matches:
                        serial_entry = {
                            "type": document_type,
                            "value": match.group(0),
                            "pattern": pattern_str,
                            "position": match.start(),
                            "validated": True,
                        }
                        # Avoid duplicates
                        if not any(
                            s["value"] == serial_entry["value"]
                            for s in serials
                        ):
                            serials.append(serial_entry)
                except re.error:
                    logger.warning(
                        "Invalid regex pattern for %s: %s",
                        document_type, pattern_str,
                    )

        # Apply generic patterns for all document types
        generic_patterns = [
            (r"(?:Ref|Reference|Ref\.)[\s:]+([A-Z0-9-/]{6,30})",
             "reference_number"),
            (r"(?:Serial|S/N|Serial No\.)[\s:]+([A-Z0-9-]{6,20})",
             "serial_number"),
            (r"(?:Control|Ctrl)[\s:]+([A-Z0-9-]{6,20})",
             "control_number"),
            (r"(?:Certificate|Cert)[\s#:]+([A-Z0-9-/]{6,30})",
             "certificate_number"),
            (r"(?:Invoice|Inv)[\s#:]+([A-Z0-9-/]{6,20})",
             "invoice_number"),
        ]

        for pattern_str, serial_type in generic_patterns:
            try:
                matches = re.finditer(pattern_str, text, re.IGNORECASE)
                for match in matches:
                    value = match.group(1).strip()
                    if len(value) >= 4:
                        serial_entry = {
                            "type": serial_type,
                            "value": value,
                            "pattern": pattern_str,
                            "position": match.start(),
                            "validated": False,
                        }
                        if not any(
                            s["value"] == serial_entry["value"]
                            for s in serials
                        ):
                            serials.append(serial_entry)
            except re.error:
                pass

        logger.debug(
            "Serial numbers extracted: count=%d", len(serials),
        )
        return serials

    # ------------------------------------------------------------------
    # Internal: Extract dates
    # ------------------------------------------------------------------

    def _extract_dates(
        self,
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Extract and normalize all date fields from metadata.

        Args:
            metadata: Merged metadata dictionary.

        Returns:
            List of dictionaries with keys: field, raw_value,
            parsed_value, source.
        """
        dates: List[Dict[str, Any]] = []
        date_fields = [
            "creation_date", "modification_date", "capture_date",
        ]

        for field in date_fields:
            raw_val = metadata.get(field)
            if raw_val:
                parsed = self._parse_date_field(raw_val)
                source = metadata.get("_metadata_sources", {}).get(
                    field, "unknown",
                )
                dates.append({
                    "field": field,
                    "raw_value": str(raw_val),
                    "parsed_value": (
                        parsed.isoformat() if parsed else None
                    ),
                    "source": source,
                })

        return dates

    # ------------------------------------------------------------------
    # Internal: Extract GPS coordinates
    # ------------------------------------------------------------------

    def _extract_gps_coordinates(
        self,
        exif_meta: Dict[str, Any],
        xmp_meta: Dict[str, Any],
        merged_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract and consolidate GPS coordinates from all sources.

        Priority: XMP GPS > EXIF GPS > text-extracted GPS.

        Args:
            exif_meta: EXIF metadata.
            xmp_meta: XMP metadata.
            merged_meta: Merged metadata.

        Returns:
            Dictionary with keys: latitude, longitude, altitude,
            source, accuracy_estimate.
        """
        gps: Dict[str, Any] = {
            "latitude": None,
            "longitude": None,
            "altitude": None,
            "source": None,
            "accuracy_estimate": None,
        }

        # Check XMP GPS (highest priority)
        if "gps_lat" in xmp_meta and "gps_lon" in xmp_meta:
            gps["latitude"] = xmp_meta["gps_lat"]
            gps["longitude"] = xmp_meta["gps_lon"]
            gps["source"] = "xmp"
            gps["accuracy_estimate"] = "xmp_embedded"

        # Check EXIF GPS
        elif "gps_lat" in exif_meta and "gps_lon" in exif_meta:
            gps["latitude"] = exif_meta["gps_lat"]
            gps["longitude"] = exif_meta["gps_lon"]
            gps["source"] = "exif"
            gps["accuracy_estimate"] = "exif_gps"
            if "gps_altitude" in exif_meta:
                gps["altitude"] = exif_meta["gps_altitude"]

        # Validate coordinate ranges
        if gps["latitude"] is not None:
            try:
                lat = float(gps["latitude"])
                lon = float(gps["longitude"])
                if not (-90.0 <= lat <= 90.0):
                    logger.warning(
                        "GPS latitude out of range: %s", lat,
                    )
                    gps["latitude"] = None
                    gps["longitude"] = None
                    gps["source"] = None
                elif not (-180.0 <= lon <= 180.0):
                    logger.warning(
                        "GPS longitude out of range: %s", lon,
                    )
                    gps["latitude"] = None
                    gps["longitude"] = None
                    gps["source"] = None
            except (ValueError, TypeError):
                gps["latitude"] = None
                gps["longitude"] = None
                gps["source"] = None

        return gps

    # ------------------------------------------------------------------
    # Internal: Check missing fields
    # ------------------------------------------------------------------

    def _check_missing_fields(
        self,
        metadata: Dict[str, Any],
        file_format: str,
    ) -> List[str]:
        """Check for required metadata fields that are missing.

        Args:
            metadata: Merged metadata dictionary.
            file_format: Detected file format.

        Returns:
            List of missing field names.
        """
        required = self._config.required_metadata_fields
        missing: List[str] = []

        for field in required:
            val = metadata.get(field)
            if val is None or val == "" or val == []:
                missing.append(field)

        return missing

    # ------------------------------------------------------------------
    # Internal: Build validation summary
    # ------------------------------------------------------------------

    def _build_validation_summary(
        self,
        anomalies: List[str],
        missing_fields: List[str],
        creation_date_anomaly: bool,
        author_match: Optional[bool],
        serial_numbers: List[Dict[str, Any]],
        gps_coords: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a summary of the validation results.

        Args:
            anomalies: List of detected anomalies.
            missing_fields: List of missing metadata fields.
            creation_date_anomaly: Whether creation date is anomalous.
            author_match: Whether author matches.
            serial_numbers: List of extracted serial numbers.
            gps_coords: GPS coordinate data.

        Returns:
            Dictionary summarizing the validation state.
        """
        # Determine overall status
        if not anomalies and not missing_fields:
            status = "clean"
        elif len(anomalies) >= 3:
            status = "high_risk"
        elif anomalies:
            status = "anomalous"
        elif missing_fields:
            status = "incomplete"
        else:
            status = "clean"

        return {
            "status": status,
            "anomaly_count": len(anomalies),
            "missing_field_count": len(missing_fields),
            "creation_date_anomaly": creation_date_anomaly,
            "author_match": author_match,
            "serial_numbers_found": len(serial_numbers),
            "gps_available": gps_coords.get("latitude") is not None,
            "risk_indicators": anomalies[:5],
        }

    # ------------------------------------------------------------------
    # Internal: Cross-validation helpers
    # ------------------------------------------------------------------

    def _cross_validate_date(
        self,
        metadata: Dict[str, Any],
        claimed_date: Any,
    ) -> Dict[str, Any]:
        """Cross-validate metadata creation date against a claimed date.

        Args:
            metadata: Extracted metadata dictionary.
            claimed_date: Claimed issuance/creation date.

        Returns:
            Dictionary with keys: match, severity, message, details.
        """
        creation_date = self._parse_date_field(
            metadata.get("creation_date"),
        )

        if not creation_date:
            return {
                "match": None,
                "severity": "warning",
                "message": (
                    "Cannot validate: creation date not found in metadata"
                ),
                "details": {"creation_date": None},
            }

        claimed_dt = self._parse_date_field(claimed_date)
        if not claimed_dt:
            return {
                "match": None,
                "severity": "warning",
                "message": "Cannot validate: claimed date is not parseable",
                "details": {"claimed_date": str(claimed_date)},
            }

        diff_days = abs((creation_date - claimed_dt).days)
        tolerance = self._config.creation_date_tolerance_days

        if creation_date > claimed_dt:
            return {
                "match": False,
                "severity": "error",
                "message": (
                    f"Creation date ({creation_date.date()}) is AFTER "
                    f"claimed date ({claimed_dt.date()})"
                ),
                "details": {
                    "creation_date": creation_date.isoformat(),
                    "claimed_date": claimed_dt.isoformat(),
                    "diff_days": diff_days,
                },
            }

        if diff_days > tolerance:
            return {
                "match": False,
                "severity": "warning",
                "message": (
                    f"Creation date ({creation_date.date()}) differs "
                    f"from claimed date ({claimed_dt.date()}) by "
                    f"{diff_days} days; exceeds {tolerance}-day tolerance"
                ),
                "details": {
                    "creation_date": creation_date.isoformat(),
                    "claimed_date": claimed_dt.isoformat(),
                    "diff_days": diff_days,
                    "tolerance_days": tolerance,
                },
            }

        return {
            "match": True,
            "severity": "info",
            "message": (
                f"Creation date matches within tolerance "
                f"(diff={diff_days} days)"
            ),
            "details": {
                "creation_date": creation_date.isoformat(),
                "claimed_date": claimed_dt.isoformat(),
                "diff_days": diff_days,
            },
        }

    def _cross_validate_author(
        self,
        metadata: Dict[str, Any],
        claimed_author: str,
    ) -> Dict[str, Any]:
        """Cross-validate metadata author against claimed author.

        Args:
            metadata: Extracted metadata dictionary.
            claimed_author: Claimed document author.

        Returns:
            Dictionary with keys: match, severity, message, details.
        """
        doc_author = metadata.get("author", "")

        if not doc_author:
            return {
                "match": None,
                "severity": "warning",
                "message": "Cannot validate: author not found in metadata",
                "details": {"metadata_author": None},
            }

        is_match = self._fuzzy_match(claimed_author, doc_author)

        if is_match:
            return {
                "match": True,
                "severity": "info",
                "message": "Author matches claimed value",
                "details": {
                    "metadata_author": doc_author,
                    "claimed_author": claimed_author,
                },
            }

        return {
            "match": False,
            "severity": "error",
            "message": (
                f"Document author '{doc_author}' does not match "
                f"claimed author '{claimed_author}'"
            ),
            "details": {
                "metadata_author": doc_author,
                "claimed_author": claimed_author,
            },
        }

    def _cross_validate_authority(
        self,
        metadata: Dict[str, Any],
        claimed_authority: str,
    ) -> Dict[str, Any]:
        """Cross-validate producer/creator against claimed authority.

        Args:
            metadata: Extracted metadata dictionary.
            claimed_authority: Claimed issuing authority.

        Returns:
            Dictionary with keys: match, severity, message, details.
        """
        producer = metadata.get("producer", "")
        creator = metadata.get("creator", "")
        author = metadata.get("author", "")

        authority_lower = claimed_authority.lower()
        fields_checked = []

        for name, val in [
            ("author", author),
            ("producer", producer),
            ("creator", creator),
        ]:
            if val:
                fields_checked.append(name)
                if authority_lower in val.lower():
                    return {
                        "match": True,
                        "severity": "info",
                        "message": (
                            f"Issuing authority found in {name}: '{val}'"
                        ),
                        "details": {
                            "field": name,
                            "field_value": val,
                            "claimed_authority": claimed_authority,
                        },
                    }

        if not fields_checked:
            return {
                "match": None,
                "severity": "warning",
                "message": (
                    "Cannot validate: no author/producer/creator in metadata"
                ),
                "details": {"claimed_authority": claimed_authority},
            }

        return {
            "match": False,
            "severity": "warning",
            "message": (
                f"Claimed authority '{claimed_authority}' not found in "
                f"metadata fields: {fields_checked}"
            ),
            "details": {
                "fields_checked": fields_checked,
                "claimed_authority": claimed_authority,
                "author": author,
                "producer": producer,
                "creator": creator,
            },
        }

    def _cross_validate_serial(
        self,
        metadata: Dict[str, Any],
        claimed_serial: str,
        document_type: Optional[str],
    ) -> Dict[str, Any]:
        """Cross-validate a claimed serial number against metadata.

        Args:
            metadata: Extracted metadata dictionary.
            claimed_serial: Claimed serial/certificate number.
            document_type: Optional document type for pattern validation.

        Returns:
            Dictionary with keys: match, severity, message, details.
        """
        # Check format validity if document type has known patterns
        if document_type and document_type in SERIAL_NUMBER_PATTERNS:
            patterns = SERIAL_NUMBER_PATTERNS[document_type]
            format_valid = any(
                re.fullmatch(p, claimed_serial) for p in patterns
            )
            if not format_valid:
                return {
                    "match": False,
                    "severity": "error",
                    "message": (
                        f"Serial number '{claimed_serial}' does not "
                        f"match expected format for '{document_type}'"
                    ),
                    "details": {
                        "claimed_serial": claimed_serial,
                        "document_type": document_type,
                        "expected_patterns": patterns,
                    },
                }

        return {
            "match": True,
            "severity": "info",
            "message": "Serial number format is valid",
            "details": {"claimed_serial": claimed_serial},
        }

    # ------------------------------------------------------------------
    # Internal: Compute author match flag
    # ------------------------------------------------------------------

    def _compute_author_match(
        self,
        metadata: Dict[str, Any],
        claimed_author: Optional[str],
    ) -> Optional[bool]:
        """Compute whether the document author matches the claimed value.

        Args:
            metadata: Merged metadata dictionary.
            claimed_author: Claimed author value.

        Returns:
            True if match, False if mismatch, None if cannot determine.
        """
        if not claimed_author:
            return None

        doc_author = metadata.get("author", "")
        if not doc_author:
            return None

        return self._fuzzy_match(claimed_author, doc_author)

    # ------------------------------------------------------------------
    # Internal: Parse date field
    # ------------------------------------------------------------------

    def _parse_date_field(
        self,
        value: Any,
    ) -> Optional[datetime]:
        """Parse a date value from various formats into a datetime.

        Handles PDF date format (D:YYYYMMDDHHmmSS), ISO 8601, and
        common date format variants.

        Args:
            value: Date string or datetime object.

        Returns:
            Parsed datetime with UTC timezone, or None if unparseable.
        """
        if value is None:
            return None

        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value

        date_str = str(value).strip()
        if not date_str:
            return None

        # Clean PDF date prefix
        if date_str.startswith("D:"):
            date_str = date_str[2:]

        # Remove timezone offset suffix for simpler parsing
        cleaned = re.sub(r"[+-]\d{2}'\d{2}'?$", "", date_str)
        cleaned = re.sub(r"Z$", "", cleaned)

        for fmt in _PDF_DATE_PATTERNS:
            # Skip formats with timezone if we already cleaned it
            if "%z" in fmt:
                continue
            clean_fmt = fmt.replace("D:", "")
            try:
                parsed = datetime.strptime(cleaned, clean_fmt)
                return parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        # Try ISO format parsing as fallback
        try:
            if "T" in date_str:
                parsed = datetime.fromisoformat(
                    date_str.replace("Z", "+00:00"),
                )
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed
        except ValueError:
            pass

        logger.debug("Could not parse date value: %s", date_str)
        return None

    # ------------------------------------------------------------------
    # Internal: Parse EXIF GPS data
    # ------------------------------------------------------------------

    def _parse_exif_gps(
        self,
        raw_bytes: bytes,
        text_content: str,
    ) -> Dict[str, Any]:
        """Parse GPS coordinates from EXIF data.

        Args:
            raw_bytes: Raw bytes for binary GPS IFD parsing.
            text_content: Latin-1 decoded text for pattern matching.

        Returns:
            Dictionary with gps_lat, gps_lon, gps_altitude if found.
        """
        gps: Dict[str, Any] = {}

        # Look for GPS DMS notation in text
        gps_matches = _GPS_DMS_PATTERN.findall(text_content)
        if len(gps_matches) >= 2:
            lat_parts = gps_matches[0]
            lon_parts = gps_matches[1]

            try:
                lat = self._dms_to_decimal(
                    float(lat_parts[0]),
                    float(lat_parts[1]),
                    float(lat_parts[2]),
                    lat_parts[3],
                )
                lon = self._dms_to_decimal(
                    float(lon_parts[0]),
                    float(lon_parts[1]),
                    float(lon_parts[2]),
                    lon_parts[3],
                )
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    gps["gps_lat"] = round(lat, 6)
                    gps["gps_lon"] = round(lon, 6)
            except (ValueError, IndexError):
                pass

        return gps

    # ------------------------------------------------------------------
    # Internal: Parse XMP GPS coordinate
    # ------------------------------------------------------------------

    def _parse_xmp_gps_coord(
        self,
        coord_str: str,
    ) -> Optional[float]:
        """Parse an XMP GPS coordinate string (DMS or decimal).

        XMP GPS format is typically: "DD,MM.MMM[NSEW]" or "DD.DDDDDD"

        Args:
            coord_str: GPS coordinate string from XMP.

        Returns:
            Decimal degrees float or None if unparseable.
        """
        if not coord_str:
            return None

        # Try decimal format
        try:
            val = float(coord_str)
            return round(val, 6)
        except ValueError:
            pass

        # Try DMS format: "DD,MM.MMMN"
        match = re.match(
            r"(\d+),(\d+\.?\d*)\s*([NSEW])", coord_str,
        )
        if match:
            try:
                degrees = float(match.group(1))
                minutes = float(match.group(2))
                direction = match.group(3)
                decimal = degrees + minutes / 60.0
                if direction in ("S", "W"):
                    decimal = -decimal
                return round(decimal, 6)
            except ValueError:
                pass

        return None

    # ------------------------------------------------------------------
    # Internal: DMS to decimal conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _dms_to_decimal(
        degrees: float,
        minutes: float,
        seconds: float,
        direction: str,
    ) -> float:
        """Convert degrees/minutes/seconds to decimal degrees.

        Args:
            degrees: Degrees component.
            minutes: Minutes component.
            seconds: Seconds component.
            direction: Cardinal direction (N, S, E, W).

        Returns:
            Decimal degrees float.
        """
        decimal = degrees + minutes / 60.0 + seconds / 3600.0
        if direction.upper() in ("S", "W"):
            decimal = -decimal
        return round(decimal, 6)

    # ------------------------------------------------------------------
    # Internal: Fuzzy string matching
    # ------------------------------------------------------------------

    @staticmethod
    def _fuzzy_match(
        expected: str,
        actual: str,
        threshold: float = 0.7,
    ) -> bool:
        """Perform fuzzy string matching between two values.

        Uses a simple normalized containment and token overlap approach
        for deterministic matching without external ML libraries.

        Args:
            expected: Expected string value.
            actual: Actual string value from metadata.
            threshold: Minimum similarity threshold (0.0-1.0).

        Returns:
            True if the strings are considered a match.
        """
        if not expected or not actual:
            return False

        exp_lower = expected.lower().strip()
        act_lower = actual.lower().strip()

        # Exact match
        if exp_lower == act_lower:
            return True

        # Containment match
        if exp_lower in act_lower or act_lower in exp_lower:
            return True

        # Token overlap
        exp_tokens = set(re.split(r"[\s,;./-]+", exp_lower))
        act_tokens = set(re.split(r"[\s,;./-]+", act_lower))

        # Remove empty tokens
        exp_tokens.discard("")
        act_tokens.discard("")

        if not exp_tokens or not act_tokens:
            return False

        overlap = len(exp_tokens & act_tokens)
        max_tokens = max(len(exp_tokens), len(act_tokens))
        similarity = overlap / max_tokens

        return similarity >= threshold

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        with self._lock:
            count = len(self._metadata_store)
        return (
            f"MetadataExtractorEngine(records={count}, "
            f"tolerance={self._config.creation_date_tolerance_days}d, "
            f"author_match={self._config.require_author_match})"
        )

    def __len__(self) -> int:
        """Return the number of stored metadata records."""
        with self._lock:
            return len(self._metadata_store)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "MetadataExtractorEngine",
    "KNOWN_PDF_PRODUCERS",
    "SERIAL_NUMBER_PATTERNS",
    "EXPECTED_METADATA_BY_FORMAT",
    "MIME_TO_FORMAT",
    "SUSPICIOUS_CREATORS",
]
