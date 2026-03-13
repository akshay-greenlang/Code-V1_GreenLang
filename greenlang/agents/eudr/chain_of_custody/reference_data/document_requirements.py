# -*- coding: utf-8 -*-
"""
Document Requirements Reference Data - AGENT-EUDR-009 Chain of Custody Agent

Provides required and optional document type definitions for each custody
event type in the EUDR Chain of Custody workflow. Enables deterministic,
zero-hallucination document completeness validation for DDS (Due Diligence
Statement) assembly without external API dependencies.

Each custody event type has a set of required and optional documents that
must be linked before the event can pass DDS completeness checks. Document
types are annotated with metadata including typical issuers, validity
periods, and tamper sensitivity flags.

Coverage:
    - 8 custody event types (transfer, export, import, processing_in,
      processing_out, storage_in, storage_out, inspection)
    - 15 document types with metadata
    - EUDR Article 4 and Annex II compliance mapping

Data Sources:
    EU Regulation 2023/1115 (EUDR) Article 4, 9, 10, 12
    EU Implementing Regulation 2024/XXX (DDS Submission Format)
    European Commission Guidance on Due Diligence (2025)
    ISO 22095:2020 Chain of Custody (General Terminology and Models)
    PRD-AGENT-EUDR-009 Appendix B

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-009 (Chain of Custody)
Agent ID: GL-EUDR-COC-009
Status: Production Ready
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data version for provenance tracking
# ---------------------------------------------------------------------------

DATA_VERSION = "1.0.0"
DATA_SOURCE = "GreenLang Reference Data v1.0.0 (2026-03)"

# ---------------------------------------------------------------------------
# Document Type Metadata
# ---------------------------------------------------------------------------
# Each document type has metadata describing its purpose, typical issuer,
# validity period, and whether it is tamper-sensitive (requires hash
# verification).


class DocumentTypeInfo:
    """Metadata for a document type used in EUDR Chain of Custody.

    Attributes:
        code: Short code for the document type.
        name: Human-readable document type name.
        description: Detailed description of the document.
        typical_issuer: Who typically issues this document.
        typical_validity_days: How long the document is valid (0 = no expiry).
        tamper_sensitive: Whether the document requires hash verification.
        eudr_article_ref: Relevant EUDR article reference.
    """

    __slots__ = (
        "code", "name", "description", "typical_issuer",
        "typical_validity_days", "tamper_sensitive", "eudr_article_ref",
    )

    def __init__(
        self,
        code: str,
        name: str,
        description: str,
        typical_issuer: str = "operator",
        typical_validity_days: int = 0,
        tamper_sensitive: bool = False,
        eudr_article_ref: str = "",
    ) -> None:
        self.code = code
        self.name = name
        self.description = description
        self.typical_issuer = typical_issuer
        self.typical_validity_days = typical_validity_days
        self.tamper_sensitive = tamper_sensitive
        self.eudr_article_ref = eudr_article_ref

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "code": self.code,
            "name": self.name,
            "description": self.description,
            "typical_issuer": self.typical_issuer,
            "typical_validity_days": self.typical_validity_days,
            "tamper_sensitive": self.tamper_sensitive,
            "eudr_article_ref": self.eudr_article_ref,
        }


# ---------------------------------------------------------------------------
# Document Type Registry (15 types)
# ---------------------------------------------------------------------------

DOCUMENT_TYPES: Dict[str, DocumentTypeInfo] = {
    "BILL_OF_LADING": DocumentTypeInfo(
        code="BILL_OF_LADING",
        name="Bill of Lading",
        description=(
            "Transport document issued by carrier acknowledging receipt "
            "of goods for shipment. Serves as receipt, contract of carriage, "
            "and document of title."
        ),
        typical_issuer="carrier",
        typical_validity_days=0,
        tamper_sensitive=True,
        eudr_article_ref="Article 9(1)(d)",
    ),
    "PHYTOSANITARY_CERT": DocumentTypeInfo(
        code="PHYTOSANITARY_CERT",
        name="Phytosanitary Certificate",
        description=(
            "Official document issued by plant protection authority "
            "certifying that consignment meets import phytosanitary "
            "requirements of the destination country."
        ),
        typical_issuer="government_authority",
        typical_validity_days=30,
        tamper_sensitive=True,
        eudr_article_ref="Article 9(1)(e)",
    ),
    "CUSTOMS_DECLARATION": DocumentTypeInfo(
        code="CUSTOMS_DECLARATION",
        name="Customs Declaration",
        description=(
            "Official document submitted to customs authorities declaring "
            "the nature, quantity, and value of imported or exported goods."
        ),
        typical_issuer="customs_broker",
        typical_validity_days=0,
        tamper_sensitive=True,
        eudr_article_ref="Article 9(1)(d)",
    ),
    "CERTIFICATE_OF_ORIGIN": DocumentTypeInfo(
        code="CERTIFICATE_OF_ORIGIN",
        name="Certificate of Origin",
        description=(
            "Document certifying the country of origin of the goods. "
            "Critical for EUDR traceability to production country."
        ),
        typical_issuer="chamber_of_commerce",
        typical_validity_days=365,
        tamper_sensitive=True,
        eudr_article_ref="Article 9(1)(b)",
    ),
    "SUSTAINABILITY_CERT": DocumentTypeInfo(
        code="SUSTAINABILITY_CERT",
        name="Sustainability Certification",
        description=(
            "Third-party certification of sustainable production practices. "
            "Includes FSC, RSPO, ISCC, UTZ, Rainforest Alliance, etc."
        ),
        typical_issuer="certification_body",
        typical_validity_days=365,
        tamper_sensitive=True,
        eudr_article_ref="Article 10(2)",
    ),
    "GEOLOCATION_DATA": DocumentTypeInfo(
        code="GEOLOCATION_DATA",
        name="Geolocation Data",
        description=(
            "GPS coordinates of production plots (latitude/longitude) as "
            "required by EUDR Article 9(1)(d). For plots >4ha, polygon "
            "boundaries are required."
        ),
        typical_issuer="operator",
        typical_validity_days=0,
        tamper_sensitive=True,
        eudr_article_ref="Article 9(1)(d)",
    ),
    "WEIGHT_CERTIFICATE": DocumentTypeInfo(
        code="WEIGHT_CERTIFICATE",
        name="Weight Certificate / Weighbridge Ticket",
        description=(
            "Official weight measurement at point of transfer. Used to "
            "verify mass balance at each custody transfer point."
        ),
        typical_issuer="weighbridge_operator",
        typical_validity_days=0,
        tamper_sensitive=True,
        eudr_article_ref="Article 9(1)(e)",
    ),
    "QUALITY_CERT": DocumentTypeInfo(
        code="QUALITY_CERT",
        name="Quality / Grading Certificate",
        description=(
            "Certificate attesting to commodity quality or grade. "
            "Includes moisture content, defect count, and grade "
            "classification relevant to yield estimation."
        ),
        typical_issuer="grading_authority",
        typical_validity_days=90,
        tamper_sensitive=False,
        eudr_article_ref="Article 9(1)(e)",
    ),
    "PURCHASE_CONTRACT": DocumentTypeInfo(
        code="PURCHASE_CONTRACT",
        name="Purchase / Sales Contract",
        description=(
            "Commercial contract documenting terms of sale, quantities, "
            "delivery conditions, and traceability obligations."
        ),
        typical_issuer="operator",
        typical_validity_days=0,
        tamper_sensitive=False,
        eudr_article_ref="Article 9(1)(a)",
    ),
    "DELIVERY_NOTE": DocumentTypeInfo(
        code="DELIVERY_NOTE",
        name="Delivery Note / Goods Received Note",
        description=(
            "Document confirming delivery of goods at destination. "
            "Records quantity received, condition, and acceptance."
        ),
        typical_issuer="receiver",
        typical_validity_days=0,
        tamper_sensitive=False,
        eudr_article_ref="Article 9(1)(d)",
    ),
    "WAREHOUSE_RECEIPT": DocumentTypeInfo(
        code="WAREHOUSE_RECEIPT",
        name="Warehouse Receipt",
        description=(
            "Document issued by warehouse operator acknowledging "
            "receipt and storage of goods. Tracks storage location, "
            "lot segregation, and condition."
        ),
        typical_issuer="warehouse_operator",
        typical_validity_days=0,
        tamper_sensitive=False,
        eudr_article_ref="Article 9(1)(d)",
    ),
    "PROCESSING_RECORD": DocumentTypeInfo(
        code="PROCESSING_RECORD",
        name="Processing / Transformation Record",
        description=(
            "Internal record documenting transformation of raw material "
            "into processed product. Tracks input/output quantities, "
            "yields, by-products, and lot traceability."
        ),
        typical_issuer="processor",
        typical_validity_days=0,
        tamper_sensitive=True,
        eudr_article_ref="Article 9(1)(e)",
    ),
    "INSPECTION_REPORT": DocumentTypeInfo(
        code="INSPECTION_REPORT",
        name="Inspection / Audit Report",
        description=(
            "Report from physical inspection or audit of goods, facility, "
            "or supply chain operations. May be from internal or third-party "
            "auditors."
        ),
        typical_issuer="auditor",
        typical_validity_days=365,
        tamper_sensitive=True,
        eudr_article_ref="Article 10(1)",
    ),
    "DDS_REFERENCE": DocumentTypeInfo(
        code="DDS_REFERENCE",
        name="Due Diligence Statement Reference",
        description=(
            "Reference number of the Due Diligence Statement submitted "
            "to the EUDR Information System. Required before placing "
            "goods on the EU market or exporting."
        ),
        typical_issuer="operator",
        typical_validity_days=0,
        tamper_sensitive=True,
        eudr_article_ref="Article 4(2)",
    ),
    "EXPORT_PERMIT": DocumentTypeInfo(
        code="EXPORT_PERMIT",
        name="Export Permit / License",
        description=(
            "Government-issued permit authorizing export of EUDR "
            "relevant commodities from the country of production. "
            "Required in countries with export controls on timber, etc."
        ),
        typical_issuer="government_authority",
        typical_validity_days=180,
        tamper_sensitive=True,
        eudr_article_ref="Article 9(1)(e)",
    ),
}

TOTAL_DOCUMENT_TYPES = len(DOCUMENT_TYPES)

# ---------------------------------------------------------------------------
# Document Requirements by Custody Event Type
# ---------------------------------------------------------------------------
# Each custody event type maps to a set of required and optional documents.
# required: Documents that MUST be linked for the event to pass completeness.
# optional: Documents that SHOULD be linked for comprehensive DDS evidence.

DOCUMENT_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "transfer": {
        "event_type": "transfer",
        "description": (
            "Physical transfer of custody between two parties (e.g., "
            "farmer to trader, trader to processor). The core EUDR "
            "traceability event."
        ),
        "required": [
            "BILL_OF_LADING",
            "WEIGHT_CERTIFICATE",
            "DELIVERY_NOTE",
            "GEOLOCATION_DATA",
        ],
        "optional": [
            "PURCHASE_CONTRACT",
            "QUALITY_CERT",
            "CERTIFICATE_OF_ORIGIN",
            "SUSTAINABILITY_CERT",
        ],
    },
    "export": {
        "event_type": "export",
        "description": (
            "Export of goods from country of production. Triggers "
            "enhanced documentation requirements including customs "
            "and phytosanitary clearances."
        ),
        "required": [
            "CUSTOMS_DECLARATION",
            "BILL_OF_LADING",
            "CERTIFICATE_OF_ORIGIN",
            "GEOLOCATION_DATA",
            "PHYTOSANITARY_CERT",
        ],
        "optional": [
            "EXPORT_PERMIT",
            "WEIGHT_CERTIFICATE",
            "SUSTAINABILITY_CERT",
            "QUALITY_CERT",
            "PURCHASE_CONTRACT",
        ],
    },
    "import": {
        "event_type": "import",
        "description": (
            "Import of goods into the EU market. Must have a valid DDS "
            "reference before customs release. EUDR Article 4 obligation."
        ),
        "required": [
            "CUSTOMS_DECLARATION",
            "BILL_OF_LADING",
            "CERTIFICATE_OF_ORIGIN",
            "GEOLOCATION_DATA",
            "DDS_REFERENCE",
        ],
        "optional": [
            "PHYTOSANITARY_CERT",
            "WEIGHT_CERTIFICATE",
            "SUSTAINABILITY_CERT",
            "QUALITY_CERT",
            "INSPECTION_REPORT",
        ],
    },
    "processing_in": {
        "event_type": "processing_in",
        "description": (
            "Receipt of raw material into a processing facility. "
            "Marks the start of a transformation step in the CoC."
        ),
        "required": [
            "WEIGHT_CERTIFICATE",
            "DELIVERY_NOTE",
            "QUALITY_CERT",
        ],
        "optional": [
            "BILL_OF_LADING",
            "GEOLOCATION_DATA",
            "SUSTAINABILITY_CERT",
            "PURCHASE_CONTRACT",
        ],
    },
    "processing_out": {
        "event_type": "processing_out",
        "description": (
            "Dispatch of processed product from a processing facility. "
            "Must include transformation records linking input to output "
            "with yield verification."
        ),
        "required": [
            "PROCESSING_RECORD",
            "WEIGHT_CERTIFICATE",
            "QUALITY_CERT",
        ],
        "optional": [
            "DELIVERY_NOTE",
            "SUSTAINABILITY_CERT",
            "INSPECTION_REPORT",
        ],
    },
    "storage_in": {
        "event_type": "storage_in",
        "description": (
            "Receipt of goods into warehouse or storage facility. "
            "Tracks lot segregation and storage conditions."
        ),
        "required": [
            "WAREHOUSE_RECEIPT",
            "WEIGHT_CERTIFICATE",
        ],
        "optional": [
            "DELIVERY_NOTE",
            "QUALITY_CERT",
            "BILL_OF_LADING",
        ],
    },
    "storage_out": {
        "event_type": "storage_out",
        "description": (
            "Dispatch of goods from warehouse or storage facility. "
            "Must reconcile with storage_in quantities for mass balance."
        ),
        "required": [
            "WAREHOUSE_RECEIPT",
            "WEIGHT_CERTIFICATE",
            "DELIVERY_NOTE",
        ],
        "optional": [
            "QUALITY_CERT",
            "BILL_OF_LADING",
            "PURCHASE_CONTRACT",
        ],
    },
    "inspection": {
        "event_type": "inspection",
        "description": (
            "Physical inspection or audit event. May be triggered by "
            "competent authority (Article 16) or voluntary third-party "
            "verification."
        ),
        "required": [
            "INSPECTION_REPORT",
        ],
        "optional": [
            "WEIGHT_CERTIFICATE",
            "QUALITY_CERT",
            "SUSTAINABILITY_CERT",
            "GEOLOCATION_DATA",
            "PROCESSING_RECORD",
        ],
    },
}

TOTAL_EVENT_TYPES = len(DOCUMENT_REQUIREMENTS)


# ---------------------------------------------------------------------------
# Accessor functions
# ---------------------------------------------------------------------------


def get_required_documents(event_type: str) -> List[str]:
    """Return the list of required document type codes for an event type.

    Args:
        event_type: Custody event type (e.g., 'transfer', 'export').

    Returns:
        List of required document type codes. Empty list if event type
        not found.

    Example:
        >>> docs = get_required_documents("transfer")
        >>> "BILL_OF_LADING" in docs
        True
    """
    req = DOCUMENT_REQUIREMENTS.get(event_type)
    if req is None:
        return []
    return list(req.get("required", []))


def get_optional_documents(event_type: str) -> List[str]:
    """Return the list of optional document type codes for an event type.

    Args:
        event_type: Custody event type.

    Returns:
        List of optional document type codes. Empty list if not found.

    Example:
        >>> docs = get_optional_documents("transfer")
        >>> "SUSTAINABILITY_CERT" in docs
        True
    """
    req = DOCUMENT_REQUIREMENTS.get(event_type)
    if req is None:
        return []
    return list(req.get("optional", []))


def is_document_required(event_type: str, document_type: str) -> bool:
    """Check whether a document type is required for an event type.

    Args:
        event_type: Custody event type.
        document_type: Document type code.

    Returns:
        True if the document is required, False otherwise.

    Example:
        >>> is_document_required("export", "CUSTOMS_DECLARATION")
        True
        >>> is_document_required("transfer", "EXPORT_PERMIT")
        False
    """
    required = get_required_documents(event_type)
    return document_type in required


def get_document_type_info(document_type: str) -> Optional[DocumentTypeInfo]:
    """Return metadata for a document type.

    Args:
        document_type: Document type code (e.g., 'BILL_OF_LADING').

    Returns:
        DocumentTypeInfo instance, or None if not found.

    Example:
        >>> info = get_document_type_info("BILL_OF_LADING")
        >>> info.typical_issuer
        'carrier'
    """
    return DOCUMENT_TYPES.get(document_type)


def get_all_document_types() -> List[str]:
    """Return all registered document type codes.

    Returns:
        Sorted list of document type codes.

    Example:
        >>> types = get_all_document_types()
        >>> len(types) >= 15
        True
    """
    return sorted(DOCUMENT_TYPES.keys())


def validate_document_completeness(
    event_type: str,
    linked_documents: List[str],
) -> Dict[str, Any]:
    """Validate document completeness for a custody event.

    Checks which required documents are present and which are missing.
    Calculates a completeness score based on required document coverage.

    Args:
        event_type: Custody event type.
        linked_documents: List of document type codes already linked
            to the event.

    Returns:
        Dictionary with: complete (bool), completeness_score (0-100),
        required_present, required_missing, optional_present,
        optional_missing.

    Example:
        >>> result = validate_document_completeness(
        ...     "transfer",
        ...     ["BILL_OF_LADING", "WEIGHT_CERTIFICATE", "DELIVERY_NOTE",
        ...      "GEOLOCATION_DATA"],
        ... )
        >>> result["complete"]
        True
    """
    req_entry = DOCUMENT_REQUIREMENTS.get(event_type)
    if req_entry is None:
        return {
            "complete": False,
            "completeness_score": 0.0,
            "event_type": event_type,
            "message": f"Unknown event type: {event_type}",
            "required_present": [],
            "required_missing": [],
            "optional_present": [],
            "optional_missing": [],
        }

    required = set(req_entry.get("required", []))
    optional = set(req_entry.get("optional", []))
    linked = set(linked_documents)

    req_present = sorted(required & linked)
    req_missing = sorted(required - linked)
    opt_present = sorted(optional & linked)
    opt_missing = sorted(optional - linked)

    # Score based on required coverage only
    if len(required) > 0:
        completeness_score = (len(req_present) / len(required)) * 100.0
    else:
        completeness_score = 100.0

    complete = len(req_missing) == 0

    return {
        "complete": complete,
        "completeness_score": round(completeness_score, 1),
        "event_type": event_type,
        "required_present": req_present,
        "required_missing": req_missing,
        "optional_present": opt_present,
        "optional_missing": opt_missing,
        "total_required": len(required),
        "total_optional": len(optional),
        "message": (
            "All required documents present."
            if complete
            else f"Missing {len(req_missing)} required document(s): "
                 f"{', '.join(req_missing)}"
        ),
    }


def get_event_types() -> List[str]:
    """Return all custody event types that have document requirements.

    Returns:
        Sorted list of event type strings.

    Example:
        >>> 'transfer' in get_event_types()
        True
    """
    return sorted(DOCUMENT_REQUIREMENTS.keys())
