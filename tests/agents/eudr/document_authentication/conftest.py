# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-012 Document Authentication Agent test suite.

Provides reusable fixtures for document records, classification results,
signature verification results, hash records, certificate chain results,
metadata records, fraud alerts, cross-reference results, helper factories,
assertion helpers, reference data constants, and engine fixtures.

Sample Documents:
    DOC_COO_COCOA_GH, DOC_FSC_CERT_BR, DOC_BOL_PALM_ID

Sample Classification Results:
    CLASSIFICATION_COO_HIGH, CLASSIFICATION_FSC_MEDIUM

Sample Signature Results:
    SIGNATURE_PADES_VALID, SIGNATURE_CADES_EXPIRED

Sample Hash Records:
    HASH_SHA256_COO, HASH_SHA512_FSC

Sample Certificate Chain Results:
    CERT_CHAIN_VALID, CERT_CHAIN_EXPIRED

Sample Metadata Records:
    METADATA_COO_FULL, METADATA_FSC_PARTIAL

Sample Fraud Alerts:
    FRAUD_ALERT_DUPLICATE, FRAUD_ALERT_QUANTITY

Sample Cross-Reference Results:
    CROSSREF_FSC_VERIFIED, CROSSREF_RSPO_EXPIRED

Helper Factories: make_document_record(), make_classification_result(),
    make_signature_result(), make_hash_record(), make_certificate_result(),
    make_metadata_record(), make_fraud_alert(), make_crossref_result()

Assertion Helpers: assert_classification_valid(), assert_signature_valid(),
    assert_hash_valid(), assert_certificate_valid(), assert_metadata_valid(),
    assert_fraud_alert_valid()

Reference Data Constants: DOCUMENT_TYPES, SIGNATURE_STANDARDS,
    SIGNATURE_STATUSES, HASH_ALGORITHMS, CERTIFICATE_STATUSES,
    FRAUD_SEVERITIES, FRAUD_PATTERN_TYPES, REGISTRY_TYPES,
    REPORT_FORMATS, CONFIDENCE_LEVELS, AUTHENTICATION_RESULTS,
    SHA256_HEX_LENGTH, SHA512_HEX_LENGTH

Engine Fixtures (8 engines with pytest.skip for unimplemented)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-012 Document Authentication Agent (GL-EUDR-DAV-012)
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHA256_HEX_LENGTH: int = 64
SHA512_HEX_LENGTH: int = 128

DOCUMENT_TYPES: List[str] = [
    "coo", "pc", "bol", "cde", "cdi",
    "rspo_cert", "fsc_cert", "iscc_cert", "ft_cert", "utz_cert",
    "ltr", "ltd", "fmp", "fc", "wqc",
    "dds_draft", "ssd", "ic", "tc", "wr",
]

SIGNATURE_STANDARDS: List[str] = [
    "cades", "pades", "xades", "jades", "qes", "pgp", "pkcs7",
]

SIGNATURE_STATUSES: List[str] = [
    "valid", "invalid", "expired", "revoked",
    "no_signature", "unknown_signer", "stripped",
]

HASH_ALGORITHMS: List[str] = ["sha256", "sha512", "hmac_sha256"]

CERTIFICATE_STATUSES: List[str] = [
    "valid", "expired", "revoked", "self_signed", "weak_key", "unknown",
]

FRAUD_SEVERITIES: List[str] = ["low", "medium", "high", "critical"]

FRAUD_PATTERN_TYPES: List[str] = [
    "duplicate_reuse",
    "quantity_tampering",
    "date_manipulation",
    "expired_cert",
    "serial_anomaly",
    "issuer_mismatch",
    "template_forgery",
    "cross_doc_inconsistency",
    "geo_impossibility",
    "velocity_anomaly",
    "modification_anomaly",
    "round_number_bias",
    "copy_paste",
    "missing_required",
    "scope_mismatch",
]

FRAUD_RULE_IDS: List[str] = [
    "FRD-001", "FRD-002", "FRD-003", "FRD-004", "FRD-005",
    "FRD-006", "FRD-007", "FRD-008", "FRD-009", "FRD-010",
    "FRD-011", "FRD-012", "FRD-013", "FRD-014", "FRD-015",
]

FRAUD_RULE_TO_PATTERN: Dict[str, str] = {
    "FRD-001": "duplicate_reuse",
    "FRD-002": "quantity_tampering",
    "FRD-003": "date_manipulation",
    "FRD-004": "expired_cert",
    "FRD-005": "serial_anomaly",
    "FRD-006": "issuer_mismatch",
    "FRD-007": "template_forgery",
    "FRD-008": "cross_doc_inconsistency",
    "FRD-009": "geo_impossibility",
    "FRD-010": "velocity_anomaly",
    "FRD-011": "modification_anomaly",
    "FRD-012": "round_number_bias",
    "FRD-013": "copy_paste",
    "FRD-014": "missing_required",
    "FRD-015": "scope_mismatch",
}

REGISTRY_TYPES: List[str] = [
    "fsc", "rspo", "iscc", "fairtrade", "utz_ra", "ippc",
]

REPORT_FORMATS: List[str] = ["json", "pdf", "csv", "eudr_xml"]

CONFIDENCE_LEVELS: List[str] = ["high", "medium", "low", "unknown"]

AUTHENTICATION_RESULTS: List[str] = [
    "authentic", "suspicious", "fraudulent", "inconclusive",
]

VERIFICATION_STATUSES: List[str] = [
    "pending", "in_progress", "completed", "failed",
]

BATCH_JOB_STATUSES: List[str] = [
    "pending", "running", "completed", "failed", "cancelled",
]

EUDR_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood",
]

DOCUMENT_LANGUAGES: List[str] = ["en", "fr", "de", "es", "pt", "id", "nl"]

METADATA_FIELDS: List[str] = [
    "title", "author", "creator", "producer",
    "creation_date", "modification_date", "keywords",
    "gps_lat", "gps_lon",
]

TRUSTED_CAS: List[str] = [
    "DigiCert Global Root G2",
    "GlobalSign Root CA - R3",
    "Entrust Root Certification Authority - G4",
    "QuoVadis Root CA 2 G3",
    "SwissSign Gold CA - G2",
    "TUEV Sued eID Root CA 1",
    "D-TRUST Root Class 3 CA 2 2009",
    "Bundesdruckerei GmbH",
]

FRAUD_SEVERITY_WEIGHTS: Dict[str, float] = {
    "low": 1.0,
    "medium": 3.0,
    "high": 7.0,
    "critical": 10.0,
}

# Registry cross-ref verification statuses
CROSSREF_STATUSES: List[str] = [
    "verified", "not_found", "expired", "revoked", "error",
]

# Minimum key sizes
MIN_RSA_KEY_SIZE: int = 2048
MIN_ECDSA_KEY_SIZE: int = 256


# ---------------------------------------------------------------------------
# Pre-generated Identifiers
# ---------------------------------------------------------------------------

# Document IDs
DOC_ID_COO_001 = "DOC-COO-GH-001"
DOC_ID_FSC_001 = "DOC-FSC-BR-001"
DOC_ID_BOL_001 = "DOC-BOL-ID-001"
DOC_ID_PC_001 = "DOC-PC-CO-001"
DOC_ID_RSPO_001 = "DOC-RSPO-MY-001"
DOC_ID_ISCC_001 = "DOC-ISCC-ID-001"

# Supplier IDs
SUP_ID_GH_001 = "SUP-GH-001"
SUP_ID_BR_001 = "SUP-BR-001"
SUP_ID_ID_001 = "SUP-ID-001"

# Shipment IDs
SHIP_ID_001 = "SHIP-2026-001"
SHIP_ID_002 = "SHIP-2026-002"
SHIP_ID_003 = "SHIP-2026-003"

# Certificate Numbers
CERT_NUM_FSC_001 = "FSC-C123456"
CERT_NUM_RSPO_001 = "RSPO-P987654"
CERT_NUM_ISCC_001 = "ISCC-EU-200-12345678"

# Report IDs
REPORT_ID_001 = "RPT-DAV-001"
REPORT_ID_002 = "RPT-DAV-002"


# ---------------------------------------------------------------------------
# Timestamp Helper
# ---------------------------------------------------------------------------

def _ts(days_ago: int = 0, hours_ago: int = 0) -> str:
    """Generate ISO timestamp relative to now."""
    return (
        datetime.now(timezone.utc) - timedelta(days=days_ago, hours=hours_ago)
    ).isoformat()


def _ts_dt(days_ago: int = 0, hours_ago: int = 0) -> datetime:
    """Generate datetime object relative to now."""
    return datetime.now(timezone.utc) - timedelta(days=days_ago, hours=hours_ago)


def _sha256(data: str) -> str:
    """Compute SHA-256 hex digest of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _sha512(data: str) -> str:
    """Compute SHA-512 hex digest of a string."""
    return hashlib.sha512(data.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Sample Document Bytes
# ---------------------------------------------------------------------------

SAMPLE_PDF_HEADER = b"%PDF-1.7\n"
SAMPLE_PDF_BYTES = SAMPLE_PDF_HEADER + b"test-content-for-eudr-doc-auth" + (b"\x00" * 100)
SAMPLE_PDF_HASH_SHA256 = hashlib.sha256(SAMPLE_PDF_BYTES).hexdigest()
SAMPLE_PDF_HASH_SHA512 = hashlib.sha512(SAMPLE_PDF_BYTES).hexdigest()

SAMPLE_EMPTY_BYTES = b""
SAMPLE_EMPTY_HASH_SHA256 = hashlib.sha256(SAMPLE_EMPTY_BYTES).hexdigest()

SAMPLE_CORRUPT_BYTES = b"\xff\xfe\xfd\xfc\xfb"

SAMPLE_LARGE_BYTES = b"EUDR-DOC-" * 100_000  # ~900 KB
SAMPLE_LARGE_HASH_SHA256 = hashlib.sha256(SAMPLE_LARGE_BYTES).hexdigest()

# Simulated PEM certificate (not real, for testing structure)
SAMPLE_CERTIFICATE_PEM = (
    "-----BEGIN CERTIFICATE-----\n"
    "MIIBkTCB+wIJALHsVZwFUfkFMA0GCSqGSIb3DQEBCwUAMBExDzANBgNVBAMMBnRl\n"
    "c3RDQTAeFw0yNjAxMDEwMDAwMDBaFw0yNzAxMDEwMDAwMDBaMBExDzANBgNVBAMM\n"
    "BnRlc3RDQTBcMA0GCSqGSIb3DQEBAQUAZEtBMAkCAgEAAgEAAgEAA0kAMEYCIQCd\n"
    "L9d1BmSBRqVJVDsPKC8JHAG0Wv8xN4yBqXYjLhAeXAIhALo+VgAWxiC1G9V2vP3Y\n"
    "-----END CERTIFICATE-----\n"
)

SAMPLE_CERTIFICATE_PEM_EXPIRED = (
    "-----BEGIN CERTIFICATE-----\n"
    "MIIBkTCB+wIJALHsVZwFUAAAAMA0GCSqGSIb3DQEBCwUAMBExDzANBgNVBAMMBnRl\n"
    "c3RDQTAeFw0yMDAxMDEwMDAwMDBaFw0yMTAxMDEwMDAwMDBaMBExDzANBgNVBAMM\n"
    "BnRlc3RDQTBcMA0GCSqGSIb3DQEBAQUAZEtBMAkCAgEAAgEAAgEAA0kAMEYCIQCd\n"
    "-----END CERTIFICATE-----\n"
)


# ---------------------------------------------------------------------------
# Sample Document Records
# ---------------------------------------------------------------------------

DOC_COO_COCOA_GH: Dict[str, Any] = {
    "document_id": DOC_ID_COO_001,
    "file_name": "certificate_of_origin_ghana_cocoa_2026.pdf",
    "file_size_bytes": 245_678,
    "file_hash_sha256": _sha256("coo-gh-cocoa-2026-content"),
    "file_hash_sha512": _sha512("coo-gh-cocoa-2026-content"),
    "document_type": "coo",
    "classification_confidence": "high",
    "language": "en",
    "commodity": "cocoa",
    "supplier_id": SUP_ID_GH_001,
    "shipment_id": SHIP_ID_001,
    "authentication_result": "authentic",
    "verification_status": "completed",
    "fraud_score": 5.0,
    "metadata": {"issuing_country": "GH", "region": "Ashanti"},
    "provenance_hash": None,
    "created_at": _ts(days_ago=30),
    "updated_at": _ts(days_ago=1),
}

DOC_FSC_CERT_BR: Dict[str, Any] = {
    "document_id": DOC_ID_FSC_001,
    "file_name": "fsc_chain_of_custody_brazil_wood.pdf",
    "file_size_bytes": 189_432,
    "file_hash_sha256": _sha256("fsc-br-wood-2026-content"),
    "file_hash_sha512": _sha512("fsc-br-wood-2026-content"),
    "document_type": "fsc_cert",
    "classification_confidence": "high",
    "language": "pt",
    "commodity": "wood",
    "supplier_id": SUP_ID_BR_001,
    "shipment_id": SHIP_ID_002,
    "authentication_result": "authentic",
    "verification_status": "completed",
    "fraud_score": 2.0,
    "metadata": {"cert_number": CERT_NUM_FSC_001, "country": "BR"},
    "provenance_hash": None,
    "created_at": _ts(days_ago=60),
    "updated_at": _ts(days_ago=5),
}

DOC_BOL_PALM_ID: Dict[str, Any] = {
    "document_id": DOC_ID_BOL_001,
    "file_name": "bill_of_lading_indonesia_palm.pdf",
    "file_size_bytes": 312_890,
    "file_hash_sha256": _sha256("bol-id-palm-2026-content"),
    "file_hash_sha512": _sha512("bol-id-palm-2026-content"),
    "document_type": "bol",
    "classification_confidence": "medium",
    "language": "id",
    "commodity": "oil_palm",
    "supplier_id": SUP_ID_ID_001,
    "shipment_id": SHIP_ID_003,
    "authentication_result": "suspicious",
    "verification_status": "completed",
    "fraud_score": 35.0,
    "metadata": {"port_of_loading": "Jakarta", "vessel": "MV Green Palm"},
    "provenance_hash": None,
    "created_at": _ts(days_ago=15),
    "updated_at": _ts(days_ago=2),
}

ALL_SAMPLE_DOCUMENTS: List[Dict[str, Any]] = [
    DOC_COO_COCOA_GH, DOC_FSC_CERT_BR, DOC_BOL_PALM_ID,
]


# ---------------------------------------------------------------------------
# Sample Classification Results
# ---------------------------------------------------------------------------

CLASSIFICATION_COO_HIGH: Dict[str, Any] = {
    "document_id": DOC_ID_COO_001,
    "document_type": "coo",
    "confidence": 0.98,
    "confidence_level": "high",
    "matched_template": "coo_ghana_standard_v2",
    "language": "en",
    "page_count": 2,
    "alternatives": [
        {"document_type": "ssd", "confidence": 0.12},
    ],
    "processing_time_ms": 145.3,
    "provenance_hash": None,
    "created_at": _ts(days_ago=30),
}

CLASSIFICATION_FSC_MEDIUM: Dict[str, Any] = {
    "document_id": DOC_ID_FSC_001,
    "document_type": "fsc_cert",
    "confidence": 0.82,
    "confidence_level": "medium",
    "matched_template": "fsc_coc_brazil_v3",
    "language": "pt",
    "page_count": 4,
    "alternatives": [
        {"document_type": "iscc_cert", "confidence": 0.35},
        {"document_type": "utz_cert", "confidence": 0.10},
    ],
    "processing_time_ms": 210.7,
    "provenance_hash": None,
    "created_at": _ts(days_ago=60),
}

ALL_SAMPLE_CLASSIFICATIONS: List[Dict[str, Any]] = [
    CLASSIFICATION_COO_HIGH, CLASSIFICATION_FSC_MEDIUM,
]


# ---------------------------------------------------------------------------
# Sample Signature Results
# ---------------------------------------------------------------------------

SIGNATURE_PADES_VALID: Dict[str, Any] = {
    "document_id": DOC_ID_COO_001,
    "signature_standard": "pades",
    "status": "valid",
    "signer_name": "Ghana Cocoa Board",
    "signer_org": "Ghana Ministry of Trade",
    "signer_email": "certs@ghanacocoa.gov.gh",
    "signing_time": _ts(days_ago=31),
    "has_timestamp": True,
    "timestamp_valid": True,
    "certificate_serial": "AB:CD:EF:12:34:56",
    "certificate_issuer": "DigiCert Global Root G2",
    "certificate_expiry": _ts(days_ago=-365),
    "multi_signature": False,
    "signature_count": 1,
    "processing_time_ms": 320.5,
    "provenance_hash": None,
    "created_at": _ts(days_ago=30),
}

SIGNATURE_CADES_EXPIRED: Dict[str, Any] = {
    "document_id": DOC_ID_BOL_001,
    "signature_standard": "cades",
    "status": "expired",
    "signer_name": "PT Kelapa Sawit",
    "signer_org": "Indonesia Palm Oil Board",
    "signer_email": "sign@ipoboard.id",
    "signing_time": _ts(days_ago=400),
    "has_timestamp": True,
    "timestamp_valid": True,
    "certificate_serial": "11:22:33:44:55:66",
    "certificate_issuer": "GlobalSign Root CA - R3",
    "certificate_expiry": _ts(days_ago=35),
    "multi_signature": False,
    "signature_count": 1,
    "processing_time_ms": 280.1,
    "provenance_hash": None,
    "created_at": _ts(days_ago=15),
}

ALL_SAMPLE_SIGNATURES: List[Dict[str, Any]] = [
    SIGNATURE_PADES_VALID, SIGNATURE_CADES_EXPIRED,
]


# ---------------------------------------------------------------------------
# Sample Hash Records
# ---------------------------------------------------------------------------

HASH_SHA256_COO: Dict[str, Any] = {
    "hash_id": f"HASH-{DOC_ID_COO_001}-SHA256",
    "document_id": DOC_ID_COO_001,
    "algorithm": "sha256",
    "hash_value": _sha256("coo-gh-cocoa-2026-content"),
    "file_size_bytes": 245_678,
    "registered_at": _ts(days_ago=30),
    "registry_status": "registered",
    "duplicate_of": None,
    "merkle_root": None,
    "provenance_hash": None,
}

HASH_SHA512_FSC: Dict[str, Any] = {
    "hash_id": f"HASH-{DOC_ID_FSC_001}-SHA512",
    "document_id": DOC_ID_FSC_001,
    "algorithm": "sha512",
    "hash_value": _sha512("fsc-br-wood-2026-content"),
    "file_size_bytes": 189_432,
    "registered_at": _ts(days_ago=60),
    "registry_status": "registered",
    "duplicate_of": None,
    "merkle_root": None,
    "provenance_hash": None,
}

ALL_SAMPLE_HASHES: List[Dict[str, Any]] = [
    HASH_SHA256_COO, HASH_SHA512_FSC,
]


# ---------------------------------------------------------------------------
# Sample Certificate Chain Results
# ---------------------------------------------------------------------------

CERT_CHAIN_VALID: Dict[str, Any] = {
    "document_id": DOC_ID_COO_001,
    "chain_length": 3,
    "leaf_subject": "Ghana Cocoa Board",
    "leaf_issuer": "DigiCert Global Root G2",
    "leaf_serial": "AB:CD:EF:12:34:56",
    "leaf_not_before": _ts(days_ago=365),
    "leaf_not_after": _ts(days_ago=-365),
    "leaf_key_type": "RSA",
    "leaf_key_size": 2048,
    "root_subject": "DigiCert Global Root G2",
    "root_trusted": True,
    "ocsp_status": "good",
    "crl_status": "not_revoked",
    "chain_status": "valid",
    "chain_certificates": [
        {"subject": "Ghana Cocoa Board", "issuer": "DigiCert SHA2 EV", "depth": 0},
        {"subject": "DigiCert SHA2 EV", "issuer": "DigiCert Global Root G2", "depth": 1},
        {"subject": "DigiCert Global Root G2", "issuer": "DigiCert Global Root G2", "depth": 2},
    ],
    "processing_time_ms": 450.2,
    "provenance_hash": None,
    "created_at": _ts(days_ago=30),
}

CERT_CHAIN_EXPIRED: Dict[str, Any] = {
    "document_id": DOC_ID_BOL_001,
    "chain_length": 2,
    "leaf_subject": "PT Kelapa Sawit",
    "leaf_issuer": "GlobalSign Root CA - R3",
    "leaf_serial": "11:22:33:44:55:66",
    "leaf_not_before": _ts(days_ago=730),
    "leaf_not_after": _ts(days_ago=35),
    "leaf_key_type": "RSA",
    "leaf_key_size": 2048,
    "root_subject": "GlobalSign Root CA - R3",
    "root_trusted": True,
    "ocsp_status": "unknown",
    "crl_status": "not_revoked",
    "chain_status": "expired",
    "chain_certificates": [
        {"subject": "PT Kelapa Sawit", "issuer": "GlobalSign Root CA - R3", "depth": 0},
        {"subject": "GlobalSign Root CA - R3", "issuer": "GlobalSign Root CA - R3", "depth": 1},
    ],
    "processing_time_ms": 380.6,
    "provenance_hash": None,
    "created_at": _ts(days_ago=15),
}

ALL_SAMPLE_CERT_CHAINS: List[Dict[str, Any]] = [
    CERT_CHAIN_VALID, CERT_CHAIN_EXPIRED,
]


# ---------------------------------------------------------------------------
# Sample Metadata Records
# ---------------------------------------------------------------------------

METADATA_COO_FULL: Dict[str, Any] = {
    "document_id": DOC_ID_COO_001,
    "title": "Certificate of Origin - Ghana Cocoa Export",
    "author": "Ghana Cocoa Board",
    "creator": "Adobe Acrobat Pro DC",
    "producer": "Adobe PDF Library 21.1.152",
    "creation_date": _ts(days_ago=32),
    "modification_date": _ts(days_ago=31),
    "keywords": "cocoa, ghana, EUDR, certificate of origin",
    "gps_lat": 6.6885,
    "gps_lon": -1.6244,
    "serial_number": "COO-GH-2026-00123",
    "issuing_authority": "Ghana Cocoa Board",
    "issuance_date": _ts(days_ago=31),
    "expiry_date": _ts(days_ago=-334),
    "page_count": 2,
    "file_format": "pdf",
    "has_embedded_images": True,
    "metadata_complete": True,
    "anomalies": [],
    "processing_time_ms": 89.4,
    "provenance_hash": None,
    "created_at": _ts(days_ago=30),
}

METADATA_FSC_PARTIAL: Dict[str, Any] = {
    "document_id": DOC_ID_FSC_001,
    "title": "FSC Chain of Custody Certificate",
    "author": None,
    "creator": "Unknown",
    "producer": "Unknown",
    "creation_date": _ts(days_ago=65),
    "modification_date": _ts(days_ago=60),
    "keywords": None,
    "gps_lat": None,
    "gps_lon": None,
    "serial_number": CERT_NUM_FSC_001,
    "issuing_authority": "FSC International",
    "issuance_date": _ts(days_ago=62),
    "expiry_date": _ts(days_ago=-300),
    "page_count": 4,
    "file_format": "pdf",
    "has_embedded_images": False,
    "metadata_complete": False,
    "anomalies": ["missing_author", "unknown_creator"],
    "processing_time_ms": 72.1,
    "provenance_hash": None,
    "created_at": _ts(days_ago=60),
}

ALL_SAMPLE_METADATA: List[Dict[str, Any]] = [
    METADATA_COO_FULL, METADATA_FSC_PARTIAL,
]


# ---------------------------------------------------------------------------
# Sample Fraud Alerts
# ---------------------------------------------------------------------------

FRAUD_ALERT_DUPLICATE: Dict[str, Any] = {
    "alert_id": "FRD-ALT-001",
    "document_id": DOC_ID_BOL_001,
    "rule_id": "FRD-001",
    "pattern_type": "duplicate_reuse",
    "severity": "high",
    "description": "Document hash matches previously submitted document DOC-BOL-ID-099",
    "evidence": {
        "original_document_id": "DOC-BOL-ID-099",
        "original_shipment_id": "SHIP-2025-999",
        "hash_match": True,
    },
    "confidence": 0.99,
    "recommended_action": "reject_and_investigate",
    "status": "open",
    "resolved_by": None,
    "resolved_at": None,
    "processing_time_ms": 15.2,
    "provenance_hash": None,
    "created_at": _ts(days_ago=2),
}

FRAUD_ALERT_QUANTITY: Dict[str, Any] = {
    "alert_id": "FRD-ALT-002",
    "document_id": DOC_ID_BOL_001,
    "rule_id": "FRD-002",
    "pattern_type": "quantity_tampering",
    "severity": "medium",
    "description": "BOL quantity 25,000 kg deviates 15% from invoice quantity 21,739 kg",
    "evidence": {
        "bol_quantity_kg": 25000.0,
        "invoice_quantity_kg": 21739.0,
        "deviation_percent": 15.0,
        "tolerance_percent": 5.0,
    },
    "confidence": 0.85,
    "recommended_action": "manual_review",
    "status": "open",
    "resolved_by": None,
    "resolved_at": None,
    "processing_time_ms": 8.7,
    "provenance_hash": None,
    "created_at": _ts(days_ago=2),
}

FRAUD_ALERT_GEO: Dict[str, Any] = {
    "alert_id": "FRD-ALT-003",
    "document_id": DOC_ID_BOL_001,
    "rule_id": "FRD-009",
    "pattern_type": "geo_impossibility",
    "severity": "critical",
    "description": "GPS coordinates place production in ocean, 200km from nearest land",
    "evidence": {
        "gps_lat": -8.123,
        "gps_lon": 115.456,
        "nearest_land_km": 200.0,
        "claimed_country": "ID",
    },
    "confidence": 0.95,
    "recommended_action": "reject_and_escalate",
    "status": "open",
    "resolved_by": None,
    "resolved_at": None,
    "processing_time_ms": 22.5,
    "provenance_hash": None,
    "created_at": _ts(days_ago=1),
}

ALL_SAMPLE_FRAUD_ALERTS: List[Dict[str, Any]] = [
    FRAUD_ALERT_DUPLICATE, FRAUD_ALERT_QUANTITY, FRAUD_ALERT_GEO,
]


# ---------------------------------------------------------------------------
# Sample Cross-Reference Results
# ---------------------------------------------------------------------------

CROSSREF_FSC_VERIFIED: Dict[str, Any] = {
    "crossref_id": "XREF-FSC-001",
    "document_id": DOC_ID_FSC_001,
    "registry_type": "fsc",
    "certificate_number": CERT_NUM_FSC_001,
    "status": "verified",
    "registry_holder": "Amazon Timber Ltd",
    "registry_scope": "wood",
    "registry_valid_from": _ts(days_ago=365),
    "registry_valid_to": _ts(days_ago=-365),
    "scope_match": True,
    "quantity_match": True,
    "party_match": True,
    "response_time_ms": 450.0,
    "cached": False,
    "cache_ttl_remaining_s": None,
    "processing_time_ms": 520.3,
    "provenance_hash": None,
    "created_at": _ts(days_ago=60),
}

CROSSREF_RSPO_EXPIRED: Dict[str, Any] = {
    "crossref_id": "XREF-RSPO-001",
    "document_id": DOC_ID_BOL_001,
    "registry_type": "rspo",
    "certificate_number": CERT_NUM_RSPO_001,
    "status": "expired",
    "registry_holder": "PT Sawit Jaya",
    "registry_scope": "oil_palm",
    "registry_valid_from": _ts(days_ago=730),
    "registry_valid_to": _ts(days_ago=30),
    "scope_match": True,
    "quantity_match": False,
    "party_match": True,
    "response_time_ms": 320.0,
    "cached": False,
    "cache_ttl_remaining_s": None,
    "processing_time_ms": 410.7,
    "provenance_hash": None,
    "created_at": _ts(days_ago=15),
}

ALL_SAMPLE_CROSSREFS: List[Dict[str, Any]] = [
    CROSSREF_FSC_VERIFIED, CROSSREF_RSPO_EXPIRED,
]


# ---------------------------------------------------------------------------
# Sample Report Data
# ---------------------------------------------------------------------------

SAMPLE_REPORT: Dict[str, Any] = {
    "report_id": REPORT_ID_001,
    "document_id": DOC_ID_COO_001,
    "report_type": "authentication",
    "format": "json",
    "classification_summary": {"type": "coo", "confidence": 0.98},
    "signature_summary": {"status": "valid", "standard": "pades"},
    "hash_summary": {"sha256": True, "sha512": True, "tampered": False},
    "certificate_summary": {"chain_valid": True, "root_trusted": True},
    "metadata_summary": {"complete": True, "anomalies": 0},
    "fraud_summary": {"alerts": 0, "risk_score": 5.0, "verdict": "low_risk"},
    "crossref_summary": {"verified": 1, "failed": 0},
    "overall_result": "authentic",
    "evidence_package_url": None,
    "retention_expires_at": _ts(days_ago=-1825),
    "provenance_hash": None,
    "created_at": _ts(days_ago=1),
}


# ---------------------------------------------------------------------------
# Helper Factories
# ---------------------------------------------------------------------------

def make_document_record(
    file_name: str = "test_document.pdf",
    document_type: str = "coo",
    commodity: str = "cocoa",
    document_id: Optional[str] = None,
    supplier_id: str = SUP_ID_GH_001,
    shipment_id: str = SHIP_ID_001,
    authentication_result: str = "authentic",
    verification_status: str = "completed",
    fraud_score: float = 0.0,
    language: str = "en",
    file_size_bytes: int = 100_000,
) -> Dict[str, Any]:
    """Build a document record dictionary for testing.

    Args:
        file_name: Original file name.
        document_type: EUDR document type identifier.
        commodity: EUDR commodity type.
        document_id: Document identifier (auto-generated if None).
        supplier_id: Supplier identifier.
        shipment_id: Shipment identifier.
        authentication_result: Overall authentication verdict.
        verification_status: Verification processing status.
        fraud_score: Composite fraud risk score (0-100).
        language: Document language code.
        file_size_bytes: File size in bytes.

    Returns:
        Dict with all document record fields.
    """
    content_seed = f"{document_type}-{commodity}-{uuid.uuid4().hex[:8]}"
    return {
        "document_id": document_id or f"DOC-{uuid.uuid4().hex[:12].upper()}",
        "file_name": file_name,
        "file_size_bytes": file_size_bytes,
        "file_hash_sha256": _sha256(content_seed),
        "file_hash_sha512": _sha512(content_seed),
        "document_type": document_type,
        "classification_confidence": "high",
        "language": language,
        "commodity": commodity,
        "supplier_id": supplier_id,
        "shipment_id": shipment_id,
        "authentication_result": authentication_result,
        "verification_status": verification_status,
        "fraud_score": fraud_score,
        "metadata": {},
        "provenance_hash": None,
        "created_at": _ts(),
        "updated_at": _ts(),
    }


def make_classification_result(
    document_id: Optional[str] = None,
    document_type: str = "coo",
    confidence: float = 0.98,
    confidence_level: str = "high",
    matched_template: Optional[str] = None,
    language: str = "en",
    page_count: int = 1,
) -> Dict[str, Any]:
    """Build a classification result dictionary for testing.

    Args:
        document_id: Document identifier (auto-generated if None).
        document_type: Classified document type.
        confidence: Classification confidence score (0.0-1.0).
        confidence_level: Confidence level (high/medium/low/unknown).
        matched_template: Template identifier that matched.
        language: Detected document language.
        page_count: Number of pages in the document.

    Returns:
        Dict with all classification result fields.
    """
    return {
        "document_id": document_id or f"DOC-{uuid.uuid4().hex[:12].upper()}",
        "document_type": document_type,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "matched_template": matched_template or f"{document_type}_default_v1",
        "language": language,
        "page_count": page_count,
        "alternatives": [],
        "processing_time_ms": 100.0,
        "provenance_hash": None,
        "created_at": _ts(),
    }


def make_signature_result(
    document_id: Optional[str] = None,
    signature_standard: str = "pades",
    status: str = "valid",
    signer_name: str = "Test Signer",
    signer_org: str = "Test Organization",
    has_timestamp: bool = True,
    certificate_issuer: str = "DigiCert Global Root G2",
) -> Dict[str, Any]:
    """Build a signature verification result dictionary for testing.

    Args:
        document_id: Document identifier (auto-generated if None).
        signature_standard: Signature standard used.
        status: Verification status.
        signer_name: Signer common name.
        signer_org: Signer organization name.
        has_timestamp: Whether a signed timestamp is present.
        certificate_issuer: Certificate issuer name.

    Returns:
        Dict with all signature verification result fields.
    """
    return {
        "document_id": document_id or f"DOC-{uuid.uuid4().hex[:12].upper()}",
        "signature_standard": signature_standard,
        "status": status,
        "signer_name": signer_name,
        "signer_org": signer_org,
        "signer_email": "test@example.com",
        "signing_time": _ts(days_ago=5),
        "has_timestamp": has_timestamp,
        "timestamp_valid": has_timestamp,
        "certificate_serial": "AA:BB:CC:DD:EE:FF",
        "certificate_issuer": certificate_issuer,
        "certificate_expiry": _ts(days_ago=-365),
        "multi_signature": False,
        "signature_count": 1,
        "processing_time_ms": 250.0,
        "provenance_hash": None,
        "created_at": _ts(),
    }


def make_hash_record(
    document_id: Optional[str] = None,
    algorithm: str = "sha256",
    hash_value: Optional[str] = None,
    file_size_bytes: int = 100_000,
    registry_status: str = "registered",
    duplicate_of: Optional[str] = None,
    hash_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a hash record dictionary for testing.

    Args:
        document_id: Document identifier (auto-generated if None).
        algorithm: Hash algorithm (sha256/sha512/hmac_sha256).
        hash_value: Hex digest (auto-generated if None).
        file_size_bytes: File size in bytes.
        registry_status: Registry status (registered/duplicate/not_found).
        duplicate_of: Document ID of duplicate if any.
        hash_id: Hash record identifier (auto-generated if None).

    Returns:
        Dict with all hash record fields.
    """
    doc_id = document_id or f"DOC-{uuid.uuid4().hex[:12].upper()}"
    if hash_value is None:
        seed = f"{doc_id}-{algorithm}-{uuid.uuid4().hex[:8]}"
        hash_value = _sha256(seed) if algorithm == "sha256" else _sha512(seed)
    return {
        "hash_id": hash_id or f"HASH-{uuid.uuid4().hex[:8].upper()}",
        "document_id": doc_id,
        "algorithm": algorithm,
        "hash_value": hash_value,
        "file_size_bytes": file_size_bytes,
        "registered_at": _ts(),
        "registry_status": registry_status,
        "duplicate_of": duplicate_of,
        "merkle_root": None,
        "provenance_hash": None,
    }


def make_certificate_result(
    document_id: Optional[str] = None,
    chain_status: str = "valid",
    chain_length: int = 3,
    leaf_subject: str = "Test Entity",
    leaf_issuer: str = "DigiCert Global Root G2",
    leaf_key_type: str = "RSA",
    leaf_key_size: int = 2048,
    root_trusted: bool = True,
    ocsp_status: str = "good",
    crl_status: str = "not_revoked",
) -> Dict[str, Any]:
    """Build a certificate chain result dictionary for testing.

    Args:
        document_id: Document identifier (auto-generated if None).
        chain_status: Overall chain validation status.
        chain_length: Number of certificates in the chain.
        leaf_subject: Subject of the leaf certificate.
        leaf_issuer: Issuer of the leaf certificate.
        leaf_key_type: Public key type (RSA/ECDSA).
        leaf_key_size: Public key size in bits.
        root_trusted: Whether the root CA is trusted.
        ocsp_status: OCSP check result.
        crl_status: CRL check result.

    Returns:
        Dict with all certificate chain result fields.
    """
    return {
        "document_id": document_id or f"DOC-{uuid.uuid4().hex[:12].upper()}",
        "chain_length": chain_length,
        "leaf_subject": leaf_subject,
        "leaf_issuer": leaf_issuer,
        "leaf_serial": "AA:BB:CC:DD",
        "leaf_not_before": _ts(days_ago=365),
        "leaf_not_after": _ts(days_ago=-365),
        "leaf_key_type": leaf_key_type,
        "leaf_key_size": leaf_key_size,
        "root_subject": leaf_issuer,
        "root_trusted": root_trusted,
        "ocsp_status": ocsp_status,
        "crl_status": crl_status,
        "chain_status": chain_status,
        "chain_certificates": [],
        "processing_time_ms": 400.0,
        "provenance_hash": None,
        "created_at": _ts(),
    }


def make_metadata_record(
    document_id: Optional[str] = None,
    title: str = "Test Document",
    author: Optional[str] = "Test Author",
    creation_date: Optional[str] = None,
    modification_date: Optional[str] = None,
    gps_lat: Optional[float] = None,
    gps_lon: Optional[float] = None,
    serial_number: Optional[str] = None,
    issuing_authority: Optional[str] = None,
    metadata_complete: bool = True,
    anomalies: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a metadata record dictionary for testing.

    Args:
        document_id: Document identifier (auto-generated if None).
        title: Document title.
        author: Document author.
        creation_date: Creation date ISO string.
        modification_date: Modification date ISO string.
        gps_lat: GPS latitude.
        gps_lon: GPS longitude.
        serial_number: Document serial number.
        issuing_authority: Issuing authority name.
        metadata_complete: Whether all required fields are present.
        anomalies: List of detected anomalies.

    Returns:
        Dict with all metadata record fields.
    """
    return {
        "document_id": document_id or f"DOC-{uuid.uuid4().hex[:12].upper()}",
        "title": title,
        "author": author,
        "creator": "Test Creator",
        "producer": "Test Producer",
        "creation_date": creation_date or _ts(days_ago=5),
        "modification_date": modification_date or _ts(days_ago=3),
        "keywords": "test, eudr, document",
        "gps_lat": gps_lat,
        "gps_lon": gps_lon,
        "serial_number": serial_number or f"SN-{uuid.uuid4().hex[:8].upper()}",
        "issuing_authority": issuing_authority or "Test Authority",
        "issuance_date": _ts(days_ago=5),
        "expiry_date": _ts(days_ago=-360),
        "page_count": 1,
        "file_format": "pdf",
        "has_embedded_images": False,
        "metadata_complete": metadata_complete,
        "anomalies": anomalies or [],
        "processing_time_ms": 50.0,
        "provenance_hash": None,
        "created_at": _ts(),
    }


def make_fraud_alert(
    document_id: Optional[str] = None,
    rule_id: str = "FRD-001",
    pattern_type: str = "duplicate_reuse",
    severity: str = "high",
    description: str = "Fraud pattern detected",
    confidence: float = 0.90,
    alert_id: Optional[str] = None,
    status: str = "open",
) -> Dict[str, Any]:
    """Build a fraud alert dictionary for testing.

    Args:
        document_id: Document identifier (auto-generated if None).
        rule_id: Fraud rule identifier (FRD-001 to FRD-015).
        pattern_type: Fraud pattern type.
        severity: Fraud severity level.
        description: Human-readable description.
        confidence: Detection confidence score.
        alert_id: Alert identifier (auto-generated if None).
        status: Alert status (open/acknowledged/resolved/dismissed).

    Returns:
        Dict with all fraud alert fields.
    """
    return {
        "alert_id": alert_id or f"FRD-ALT-{uuid.uuid4().hex[:8].upper()}",
        "document_id": document_id or f"DOC-{uuid.uuid4().hex[:12].upper()}",
        "rule_id": rule_id,
        "pattern_type": pattern_type,
        "severity": severity,
        "description": description,
        "evidence": {},
        "confidence": confidence,
        "recommended_action": "manual_review",
        "status": status,
        "resolved_by": None,
        "resolved_at": None,
        "processing_time_ms": 10.0,
        "provenance_hash": None,
        "created_at": _ts(),
    }


def make_crossref_result(
    document_id: Optional[str] = None,
    registry_type: str = "fsc",
    certificate_number: Optional[str] = None,
    status: str = "verified",
    scope_match: bool = True,
    quantity_match: bool = True,
    party_match: bool = True,
    cached: bool = False,
    crossref_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a cross-reference result dictionary for testing.

    Args:
        document_id: Document identifier (auto-generated if None).
        registry_type: External registry type.
        certificate_number: Certificate number to verify.
        status: Verification status.
        scope_match: Whether the scope matches.
        quantity_match: Whether quantity matches.
        party_match: Whether the party matches.
        cached: Whether result was served from cache.
        crossref_id: Cross-reference identifier (auto-generated if None).

    Returns:
        Dict with all cross-reference result fields.
    """
    return {
        "crossref_id": crossref_id or f"XREF-{uuid.uuid4().hex[:8].upper()}",
        "document_id": document_id or f"DOC-{uuid.uuid4().hex[:12].upper()}",
        "registry_type": registry_type,
        "certificate_number": certificate_number or f"CERT-{uuid.uuid4().hex[:8].upper()}",
        "status": status,
        "registry_holder": "Test Holder",
        "registry_scope": "wood",
        "registry_valid_from": _ts(days_ago=365),
        "registry_valid_to": _ts(days_ago=-365),
        "scope_match": scope_match,
        "quantity_match": quantity_match,
        "party_match": party_match,
        "response_time_ms": 300.0,
        "cached": cached,
        "cache_ttl_remaining_s": 3600 if cached else None,
        "processing_time_ms": 350.0,
        "provenance_hash": None,
        "created_at": _ts(),
    }


# ---------------------------------------------------------------------------
# Assertion Helpers
# ---------------------------------------------------------------------------

def compute_sha256(data: Any) -> str:
    """Compute SHA-256 hash of data for provenance verification.

    Args:
        data: Data to hash (will be JSON-serialized).

    Returns:
        64-character hex digest string.
    """
    if isinstance(data, str):
        payload = data
    elif isinstance(data, bytes):
        return hashlib.sha256(data).hexdigest()
    else:
        payload = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_valid_provenance_hash(hash_value: str) -> None:
    """Assert that a provenance hash is a valid SHA-256 hex digest.

    Args:
        hash_value: The hash string to validate.

    Raises:
        AssertionError: If hash is not a valid 64-char hex string.
    """
    assert isinstance(hash_value, str), f"Hash must be string, got {type(hash_value)}"
    assert len(hash_value) == SHA256_HEX_LENGTH, (
        f"Hash length must be {SHA256_HEX_LENGTH}, got {len(hash_value)}"
    )
    assert all(c in "0123456789abcdef" for c in hash_value), (
        "Hash must be lowercase hex characters only"
    )


def assert_classification_valid(result: Dict[str, Any]) -> None:
    """Assert that a classification result has all required fields and valid values.

    Args:
        result: Classification result dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "document_id" in result, "Missing document_id"
    assert "document_type" in result, "Missing document_type"
    assert result["document_type"] in DOCUMENT_TYPES, (
        f"Invalid document_type: {result['document_type']}"
    )
    assert "confidence" in result, "Missing confidence"
    assert 0.0 <= result["confidence"] <= 1.0, (
        f"Confidence must be in [0,1], got {result['confidence']}"
    )
    assert "confidence_level" in result, "Missing confidence_level"
    assert result["confidence_level"] in CONFIDENCE_LEVELS, (
        f"Invalid confidence_level: {result['confidence_level']}"
    )


def assert_signature_valid(result: Dict[str, Any]) -> None:
    """Assert that a signature verification result has valid structure.

    Args:
        result: Signature result dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "document_id" in result, "Missing document_id"
    assert "signature_standard" in result, "Missing signature_standard"
    assert result["signature_standard"] in SIGNATURE_STANDARDS, (
        f"Invalid signature_standard: {result['signature_standard']}"
    )
    assert "status" in result, "Missing status"
    assert result["status"] in SIGNATURE_STATUSES, (
        f"Invalid status: {result['status']}"
    )


def assert_hash_valid(record: Dict[str, Any]) -> None:
    """Assert that a hash record has valid structure.

    Args:
        record: Hash record dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "document_id" in record, "Missing document_id"
    assert "algorithm" in record, "Missing algorithm"
    assert record["algorithm"] in HASH_ALGORITHMS, (
        f"Invalid algorithm: {record['algorithm']}"
    )
    assert "hash_value" in record, "Missing hash_value"
    hv = record["hash_value"]
    if record["algorithm"] == "sha256":
        assert len(hv) == SHA256_HEX_LENGTH, f"SHA-256 hash must be 64 chars, got {len(hv)}"
    elif record["algorithm"] == "sha512":
        assert len(hv) == SHA512_HEX_LENGTH, f"SHA-512 hash must be 128 chars, got {len(hv)}"


def assert_certificate_valid(result: Dict[str, Any]) -> None:
    """Assert that a certificate chain result has valid structure.

    Args:
        result: Certificate chain result dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "document_id" in result, "Missing document_id"
    assert "chain_status" in result, "Missing chain_status"
    assert result["chain_status"] in CERTIFICATE_STATUSES, (
        f"Invalid chain_status: {result['chain_status']}"
    )
    assert "chain_length" in result, "Missing chain_length"
    assert result["chain_length"] >= 0, "chain_length must be non-negative"


def assert_metadata_valid(record: Dict[str, Any]) -> None:
    """Assert that a metadata record has valid structure.

    Args:
        record: Metadata record dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "document_id" in record, "Missing document_id"
    assert "metadata_complete" in record, "Missing metadata_complete"
    assert isinstance(record["metadata_complete"], bool), "metadata_complete must be bool"
    assert "anomalies" in record, "Missing anomalies"
    assert isinstance(record["anomalies"], list), "anomalies must be list"


def assert_fraud_alert_valid(alert: Dict[str, Any]) -> None:
    """Assert that a fraud alert has valid structure.

    Args:
        alert: Fraud alert dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "alert_id" in alert, "Missing alert_id"
    assert "document_id" in alert, "Missing document_id"
    assert "rule_id" in alert, "Missing rule_id"
    assert "pattern_type" in alert, "Missing pattern_type"
    assert alert["pattern_type"] in FRAUD_PATTERN_TYPES, (
        f"Invalid pattern_type: {alert['pattern_type']}"
    )
    assert "severity" in alert, "Missing severity"
    assert alert["severity"] in FRAUD_SEVERITIES, (
        f"Invalid severity: {alert['severity']}"
    )
    assert "confidence" in alert, "Missing confidence"
    assert 0.0 <= alert["confidence"] <= 1.0, (
        f"Confidence must be in [0,1], got {alert['confidence']}"
    )


def assert_crossref_valid(result: Dict[str, Any]) -> None:
    """Assert that a cross-reference result has valid structure.

    Args:
        result: Cross-reference result dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "document_id" in result, "Missing document_id"
    assert "registry_type" in result, "Missing registry_type"
    assert result["registry_type"] in REGISTRY_TYPES + ["national_customs"], (
        f"Invalid registry_type: {result['registry_type']}"
    )
    assert "status" in result, "Missing status"
    assert result["status"] in CROSSREF_STATUSES, (
        f"Invalid status: {result['status']}"
    )


def assert_valid_score(score: float, min_val: float = 0.0, max_val: float = 100.0) -> None:
    """Assert that a score is within valid bounds.

    Args:
        score: The score value to validate.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.

    Raises:
        AssertionError: If score is out of bounds.
    """
    assert isinstance(score, (int, float)), f"Score must be numeric, got {type(score)}"
    assert min_val <= score <= max_val, f"Score {score} out of bounds [{min_val}, {max_val}]"


# ---------------------------------------------------------------------------
# Configuration Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def dav_config() -> Dict[str, Any]:
    """Create a DocumentAuthenticationConfig-compatible dictionary with test defaults."""
    return {
        "database_url": "postgresql://localhost:5432/greenlang_test",
        "redis_url": "redis://localhost:6379/1",
        "log_level": "DEBUG",
        "pool_size": 5,
        "min_confidence_high": 0.95,
        "min_confidence_medium": 0.70,
        "max_batch_size": 500,
        "signature_timeout_s": 30,
        "require_timestamp": True,
        "accept_self_signed": False,
        "hash_algorithm": "sha256",
        "secondary_hash": "sha512",
        "registry_ttl_days": 1825,
        "ocsp_enabled": True,
        "crl_refresh_hours": 24,
        "min_key_size_rsa": 2048,
        "min_key_size_ecdsa": 256,
        "cert_transparency_enabled": False,
        "creation_date_tolerance_days": 30,
        "require_author_match": True,
        "flag_empty_metadata": True,
        "quantity_tolerance_percent": 5.0,
        "date_tolerance_days": 30,
        "velocity_threshold_per_day": 10,
        "round_number_threshold_percent": 80.0,
        "fraud_rules_enabled": True,
        "cache_ttl_hours": 24,
        "fsc_api_rate_limit": 100,
        "rspo_api_rate_limit": 60,
        "iscc_api_rate_limit": 30,
        "crossref_timeout_s": 30,
        "default_format": "json",
        "retention_days": 1825,
        "evidence_package_enabled": True,
        "batch_max_size": 500,
        "batch_concurrency": 4,
        "batch_timeout_s": 300,
        "retention_years": 5,
        "eudr_commodities": list(EUDR_COMMODITIES),
        "document_types": list(DOCUMENT_TYPES),
        "fraud_pattern_types": list(FRAUD_PATTERN_TYPES),
        "trusted_cas": list(TRUSTED_CAS),
        "registry_rate_limits": {
            "fsc": 100, "rspo": 60, "iscc": 30,
            "fairtrade": 30, "utz_ra": 30, "ippc": 20,
        },
        "required_metadata_fields": ["title", "author", "creation_date", "producer"],
        "fraud_severity_weights": dict(FRAUD_SEVERITY_WEIGHTS),
        "enable_provenance": True,
        "genesis_hash": "GL-EUDR-DAV-012-TEST-GENESIS",
        "enable_metrics": False,
        "rate_limit": 300,
    }


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset any singleton state between tests to prevent cross-test contamination."""
    yield
    try:
        from greenlang.agents.eudr.document_authentication.config import reset_config
        reset_config()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Engine Fixtures (with graceful pytest.skip for unimplemented)
# ---------------------------------------------------------------------------

@pytest.fixture
def classifier_engine(dav_config):
    """Create a DocumentClassifierEngine instance for testing."""
    try:
        from greenlang.agents.eudr.document_authentication.document_classifier import (
            DocumentClassifierEngine,
        )
        return DocumentClassifierEngine(config=dav_config)
    except ImportError:
        pytest.skip("DocumentClassifierEngine not yet implemented")


@pytest.fixture
def signature_engine(dav_config):
    """Create a SignatureVerifierEngine instance for testing."""
    try:
        from greenlang.agents.eudr.document_authentication.signature_verifier import (
            SignatureVerifierEngine,
        )
        return SignatureVerifierEngine(config=dav_config)
    except ImportError:
        pytest.skip("SignatureVerifierEngine not yet implemented")


@pytest.fixture
def hash_engine(dav_config):
    """Create a HashIntegrityValidator instance for testing."""
    try:
        from greenlang.agents.eudr.document_authentication.hash_integrity_validator import (
            HashIntegrityValidator,
        )
        return HashIntegrityValidator(config=dav_config)
    except ImportError:
        pytest.skip("HashIntegrityValidator not yet implemented")


@pytest.fixture
def certificate_engine(dav_config):
    """Create a CertificateChainValidator instance for testing."""
    try:
        from greenlang.agents.eudr.document_authentication.certificate_chain_validator import (
            CertificateChainValidator,
        )
        return CertificateChainValidator(config=dav_config)
    except ImportError:
        pytest.skip("CertificateChainValidator not yet implemented")


@pytest.fixture
def metadata_engine(dav_config):
    """Create a MetadataExtractorEngine instance for testing."""
    try:
        from greenlang.agents.eudr.document_authentication.metadata_extractor import (
            MetadataExtractorEngine,
        )
        return MetadataExtractorEngine(config=dav_config)
    except ImportError:
        pytest.skip("MetadataExtractorEngine not yet implemented")


@pytest.fixture
def fraud_engine(dav_config):
    """Create a FraudPatternDetector instance for testing."""
    try:
        from greenlang.agents.eudr.document_authentication.fraud_pattern_detector import (
            FraudPatternDetector,
        )
        return FraudPatternDetector(config=dav_config)
    except ImportError:
        pytest.skip("FraudPatternDetector not yet implemented")


@pytest.fixture
def crossref_engine(dav_config):
    """Create a CrossReferenceVerifier instance for testing."""
    try:
        from greenlang.agents.eudr.document_authentication.cross_reference_verifier import (
            CrossReferenceVerifier,
        )
        return CrossReferenceVerifier(config=dav_config)
    except ImportError:
        pytest.skip("CrossReferenceVerifier not yet implemented")


@pytest.fixture
def reporter_engine(dav_config):
    """Create a ComplianceReporter instance for testing."""
    try:
        from greenlang.agents.eudr.document_authentication.compliance_reporter import (
            ComplianceReporter,
        )
        return ComplianceReporter(config=dav_config)
    except ImportError:
        pytest.skip("ComplianceReporter not yet implemented")


@pytest.fixture
def service_fixture(dav_config):
    """Create the top-level DocumentAuthenticationService facade for testing."""
    try:
        from greenlang.agents.eudr.document_authentication.setup import (
            DocumentAuthenticationService,
        )
        return DocumentAuthenticationService(config=dav_config)
    except ImportError:
        pytest.skip("DocumentAuthenticationService not yet implemented")


# ---------------------------------------------------------------------------
# Data Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_document() -> Dict[str, Any]:
    """Return a sample COO document record."""
    return copy.deepcopy(DOC_COO_COCOA_GH)


@pytest.fixture
def sample_classification() -> Dict[str, Any]:
    """Return a sample high-confidence classification result."""
    return copy.deepcopy(CLASSIFICATION_COO_HIGH)


@pytest.fixture
def sample_signature() -> Dict[str, Any]:
    """Return a sample valid PAdES signature result."""
    return copy.deepcopy(SIGNATURE_PADES_VALID)


@pytest.fixture
def sample_hash() -> Dict[str, Any]:
    """Return a sample SHA-256 hash record."""
    return copy.deepcopy(HASH_SHA256_COO)


@pytest.fixture
def sample_cert_chain() -> Dict[str, Any]:
    """Return a sample valid certificate chain result."""
    return copy.deepcopy(CERT_CHAIN_VALID)


@pytest.fixture
def sample_metadata() -> Dict[str, Any]:
    """Return a sample full metadata record."""
    return copy.deepcopy(METADATA_COO_FULL)


@pytest.fixture
def sample_fraud_alert() -> Dict[str, Any]:
    """Return a sample duplicate reuse fraud alert."""
    return copy.deepcopy(FRAUD_ALERT_DUPLICATE)


@pytest.fixture
def sample_crossref() -> Dict[str, Any]:
    """Return a sample verified FSC cross-reference result."""
    return copy.deepcopy(CROSSREF_FSC_VERIFIED)


@pytest.fixture
def sample_report() -> Dict[str, Any]:
    """Return a sample authentication report."""
    return copy.deepcopy(SAMPLE_REPORT)


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Return sample PDF bytes for testing."""
    return SAMPLE_PDF_BYTES


@pytest.fixture
def sample_certificate_pem() -> str:
    """Return sample PEM certificate string for testing."""
    return SAMPLE_CERTIFICATE_PEM


@pytest.fixture
def sample_document_metadata() -> Dict[str, Any]:
    """Return sample document metadata context for testing."""
    return {
        "issuing_country": "GH",
        "commodity": "cocoa",
        "supplier_id": SUP_ID_GH_001,
        "shipment_id": SHIP_ID_001,
        "declared_quantity_kg": 25000.0,
        "declared_date": _ts(days_ago=30),
    }


@pytest.fixture
def sample_fraud_context() -> Dict[str, Any]:
    """Return sample fraud detection context for testing."""
    return {
        "document_id": DOC_ID_BOL_001,
        "document_type": "bol",
        "commodity": "oil_palm",
        "supplier_id": SUP_ID_ID_001,
        "shipment_id": SHIP_ID_003,
        "declared_quantity_kg": 25000.0,
        "invoice_quantity_kg": 21739.0,
        "gps_lat": -6.2,
        "gps_lon": 106.8,
        "creation_date": _ts(days_ago=15),
        "modification_date": _ts(days_ago=10),
        "issuance_date": _ts(days_ago=14),
        "serial_number": "BOL-ID-2026-XYZ",
        "previous_documents": [DOC_ID_COO_001, DOC_ID_FSC_001],
    }


@pytest.fixture
def sample_crossref_context() -> Dict[str, Any]:
    """Return sample cross-reference context for testing."""
    return {
        "document_id": DOC_ID_FSC_001,
        "registry_type": "fsc",
        "certificate_number": CERT_NUM_FSC_001,
        "claimed_holder": "Amazon Timber Ltd",
        "claimed_scope": "wood",
        "claimed_quantity_kg": 50000.0,
    }


# ---------------------------------------------------------------------------
# Parametrized Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=DOCUMENT_TYPES)
def document_type(request) -> str:
    """Parametrize across all 20 EUDR document types."""
    return request.param


@pytest.fixture(params=SIGNATURE_STANDARDS)
def signature_standard(request) -> str:
    """Parametrize across all 7 signature standards."""
    return request.param


@pytest.fixture(params=FRAUD_PATTERN_TYPES)
def fraud_pattern(request) -> str:
    """Parametrize across all 15 fraud pattern types."""
    return request.param


@pytest.fixture(params=REGISTRY_TYPES)
def registry_type(request) -> str:
    """Parametrize across all 6 registry types."""
    return request.param


@pytest.fixture(params=REPORT_FORMATS)
def report_format(request) -> str:
    """Parametrize across all 4 report formats."""
    return request.param


@pytest.fixture(params=EUDR_COMMODITIES)
def commodity(request) -> str:
    """Parametrize across all 7 EUDR commodities."""
    return request.param


@pytest.fixture(params=CONFIDENCE_LEVELS)
def confidence_level(request) -> str:
    """Parametrize across all 4 confidence levels."""
    return request.param


@pytest.fixture(params=SIGNATURE_STATUSES)
def sig_status(request) -> str:
    """Parametrize across all 7 signature statuses."""
    return request.param


@pytest.fixture(params=CERTIFICATE_STATUSES)
def cert_status(request) -> str:
    """Parametrize across all 6 certificate statuses."""
    return request.param


@pytest.fixture(params=FRAUD_SEVERITIES)
def fraud_severity(request) -> str:
    """Parametrize across all 4 fraud severities."""
    return request.param


@pytest.fixture(params=CROSSREF_STATUSES)
def crossref_status(request) -> str:
    """Parametrize across all 5 cross-reference statuses."""
    return request.param
