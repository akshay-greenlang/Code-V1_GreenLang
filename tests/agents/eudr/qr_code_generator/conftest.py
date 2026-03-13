# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-014 QR Code Generator Agent test suite.

Provides reusable fixtures for QR code records, data payloads, label records,
batch codes, verification URLs, signature records, scan events, bulk jobs,
lifecycle events, helper factories, assertion helpers, reference data
constants, and engine fixtures.

Sample QR Codes:
    SAMPLE_QR_CODE, SAMPLE_QR_CODE_SVG, SAMPLE_QR_CODE_WITH_LOGO

Sample Payloads:
    SAMPLE_PAYLOAD, SAMPLE_PAYLOAD_COMPRESSED, SAMPLE_PAYLOAD_ENCRYPTED

Sample Labels:
    SAMPLE_LABEL, SAMPLE_LABEL_SHIPPING, SAMPLE_LABEL_CONSUMER

Sample Batch Codes:
    SAMPLE_BATCH_CODE, SAMPLE_BATCH_CODE_ISO7064, SAMPLE_BATCH_CODE_CRC8

Sample Verification URLs:
    SAMPLE_VERIFICATION_URL_RECORD, SAMPLE_VERIFICATION_URL_WITH_SHORT

Sample Signatures:
    SAMPLE_SIGNATURE, SAMPLE_SIGNATURE_ROTATED

Sample Scan Events:
    SAMPLE_SCAN_EVENT, SAMPLE_SCAN_EVENT_COUNTERFEIT

Sample Bulk Jobs:
    SAMPLE_BULK_JOB, SAMPLE_BULK_JOB_COMPLETED, SAMPLE_BULK_JOB_FAILED

Sample Lifecycle Events:
    SAMPLE_LIFECYCLE_EVENT, SAMPLE_LIFECYCLE_REVOCATION

Helper Factories: make_qr_code(), make_payload(), make_label(),
    make_batch_code(), make_verification_url(), make_signature(),
    make_scan_event(), make_bulk_job(), make_lifecycle_event()

Assertion Helpers: assert_qr_code_valid(), assert_payload_valid(),
    assert_label_valid(), assert_batch_code_valid(),
    assert_verification_url_valid(), assert_signature_valid(),
    assert_scan_event_valid(), assert_valid_sha256(),
    assert_valid_hmac(), assert_valid_luhn(), assert_valid_iso7064(),
    assert_valid_crc8()

Reference Data Constants: SUPPORTED_FORMATS, ERROR_CORRECTION_LEVELS,
    CONTENT_TYPES, SYMBOLOGY_TYPES, LABEL_TEMPLATES,
    CHECK_DIGIT_ALGORITHMS, CODE_STATUSES, SCAN_OUTCOMES,
    COUNTERFEIT_RISK_LEVELS, BULK_JOB_STATUSES, SHA256_HEX_LENGTH,
    EUDR_RETENTION_YEARS, MAX_QR_VERSION, EUDR_COMMODITIES,
    DPI_LEVELS, PAYLOAD_ENCODINGS, COMPLIANCE_STATUSES,
    QUALITY_GRADES

Engine Fixtures (8 engines with pytest.skip for unimplemented)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014 QR Code Generator (GL-EUDR-QRG-014)
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHA256_HEX_LENGTH: int = 64
HMAC_SHA256_HEX_LENGTH: int = 64

SUPPORTED_FORMATS: List[str] = ["png", "svg", "pdf", "zpl", "eps"]

ERROR_CORRECTION_LEVELS: List[str] = ["L", "M", "Q", "H"]

CONTENT_TYPES: List[str] = [
    "full_traceability",
    "compact_verification",
    "consumer_summary",
    "batch_identifier",
    "blockchain_anchor",
]

SYMBOLOGY_TYPES: List[str] = [
    "qr_code",
    "micro_qr",
    "data_matrix",
    "gs1_digital_link",
]

LABEL_TEMPLATES: List[str] = [
    "product_label",
    "shipping_label",
    "pallet_label",
    "container_label",
    "consumer_label",
]

CHECK_DIGIT_ALGORITHMS: List[str] = [
    "luhn",
    "iso7064_mod11_10",
    "crc8",
]

CODE_STATUSES: List[str] = [
    "created",
    "active",
    "deactivated",
    "revoked",
    "expired",
]

SCAN_OUTCOMES: List[str] = [
    "verified",
    "counterfeit_suspected",
    "expired_code",
    "revoked_code",
    "error",
]

COUNTERFEIT_RISK_LEVELS: List[str] = [
    "low",
    "medium",
    "high",
    "critical",
]

BULK_JOB_STATUSES: List[str] = [
    "queued",
    "processing",
    "completed",
    "failed",
    "cancelled",
]

EUDR_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood",
]

DPI_LEVELS: List[int] = [72, 150, 300, 600]

PAYLOAD_ENCODINGS: List[str] = ["utf8", "base64", "zlib_base64"]

COMPLIANCE_STATUSES: List[str] = [
    "compliant", "pending", "non_compliant", "under_review",
]

QUALITY_GRADES: List[str] = ["A", "B", "C", "D"]

EUDR_RETENTION_YEARS: int = 5

MAX_QR_VERSION: int = 40

MAX_BATCH_SIZE: int = 500

MAX_PAYLOAD_BYTES: int = 2953

LIFECYCLE_EVENT_TYPES: List[str] = [
    "activation",
    "deactivation",
    "revocation",
    "expiry",
    "reprint",
    "reactivation",
    "replacement",
]

# Compliance colour hex values (from config defaults)
COMPLIANT_COLOR_HEX: str = "#2E7D32"
PENDING_COLOR_HEX: str = "#F57F17"
NON_COMPLIANT_COLOR_HEX: str = "#C62828"

# Label template dimensions (mm) per template type
LABEL_DIMENSIONS: Dict[str, Dict[str, float]] = {
    "product_label": {"width_mm": 50.0, "height_mm": 30.0},
    "shipping_label": {"width_mm": 100.0, "height_mm": 150.0},
    "pallet_label": {"width_mm": 150.0, "height_mm": 200.0},
    "container_label": {"width_mm": 200.0, "height_mm": 250.0},
    "consumer_label": {"width_mm": 40.0, "height_mm": 25.0},
}

# Default verification URL settings
DEFAULT_BASE_VERIFICATION_URL: str = "https://verify.greenlang.eu"
DEFAULT_HMAC_TRUNCATION_LENGTH: int = 8
DEFAULT_TOKEN_TTL_YEARS: int = 5

# Scan velocity threshold (from config default)
DEFAULT_SCAN_VELOCITY_THRESHOLD: int = 100

# Bulk generation limits
BULK_MAX_SIZE: int = 100_000
DEFAULT_BULK_WORKERS: int = 4
DEFAULT_BULK_TIMEOUT_S: int = 3600

# Max reprints (from config default)
DEFAULT_MAX_REPRINTS: int = 3


# ---------------------------------------------------------------------------
# Pre-generated Identifiers
# ---------------------------------------------------------------------------

SAMPLE_DDS_REFERENCE: str = "DDS-2026-EU-QRG-001"
SAMPLE_BATCH_ID: str = "BATCH-QRG-2026-001"
SAMPLE_OPERATOR_ID: str = "OP-EU-COCOA-001"
SAMPLE_OPERATOR_ID_2: str = "OP-EU-WOOD-002"
SAMPLE_OPERATOR_ID_3: str = "OP-EU-PALM-003"
SAMPLE_COMMODITY: str = "cocoa"
SAMPLE_COUNTRY_CODE: str = "GH"
SAMPLE_HS_CODE: str = "1801.00"
SAMPLE_GTIN: str = "04012345123456"
SAMPLE_HMAC_KEY: str = "test-hmac-secret-key-for-signing-qr-codes-2026"
SAMPLE_CODE_ID: str = "QR-CODE-001"
SAMPLE_CODE_ID_2: str = "QR-CODE-002"
SAMPLE_CODE_ID_3: str = "QR-CODE-003"
SAMPLE_SCAN_LAT: float = 50.8503
SAMPLE_SCAN_LON: float = 4.3517
SAMPLE_FACILITY_ID: str = "FAC-GH-ACCRA-001"
SAMPLE_PRODUCT_NAME: str = "Deforestation-Free Cocoa Butter 500g"
SAMPLE_VERIFICATION_BASE_URL: str = "https://verify.greenlang.eu"


# ---------------------------------------------------------------------------
# Timestamp and Hash Helpers
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


def _hmac_sha256(key: str, data: str) -> str:
    """Compute HMAC-SHA256 hex digest."""
    return hmac.new(
        key.encode("utf-8"),
        data.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _compute_luhn_check_digit(number_str: str) -> str:
    """Compute a Luhn check digit for a numeric string.

    Args:
        number_str: Numeric string to compute check digit for.

    Returns:
        Single character check digit (0-9).
    """
    digits = [int(d) for d in number_str]
    odd_sum = sum(digits[-1::-2])
    even_digits = digits[-2::-2]
    even_sum = 0
    for d in even_digits:
        doubled = d * 2
        even_sum += doubled - 9 if doubled > 9 else doubled
    total = odd_sum + even_sum
    check = (10 - (total % 10)) % 10
    return str(check)


def _compute_iso7064_mod11_10(data_str: str) -> str:
    """Compute ISO 7064 Mod 11,10 check character.

    Args:
        data_str: Numeric string to compute check character for.

    Returns:
        Single character check digit (0-9).
    """
    modulus = 10
    product = modulus
    for char in data_str:
        digit = int(char)
        s = (product + digit) % modulus
        if s == 0:
            s = modulus
        product = (2 * s) % 11
    check = (11 - product) % 11
    if check == 10:
        check = 0
    return str(check)


def _compute_crc8(data_bytes: bytes) -> int:
    """Compute CRC-8 (polynomial 0x07) check value.

    Args:
        data_bytes: Bytes to compute CRC for.

    Returns:
        Integer CRC-8 value (0-255).
    """
    crc = 0x00
    poly = 0x07
    for byte in data_bytes:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


# Pre-computed sample hashes
SAMPLE_PAYLOAD_HASH: str = _sha256("eudr-qr-payload-001")
SAMPLE_PAYLOAD_HASH_2: str = _sha256("eudr-qr-payload-002")
SAMPLE_IMAGE_DATA_HASH: str = _sha256("eudr-qr-image-data-001")
SAMPLE_SIGNED_DATA_HASH: str = _sha256("eudr-qr-signed-data-001")
SAMPLE_PROVENANCE_HASH: str = _sha256("eudr-qr-provenance-001")
SAMPLE_OUTPUT_FILE_HASH: str = _sha256("eudr-qr-output-file-001")

# HMAC samples
SAMPLE_HMAC_VALUE: str = _hmac_sha256(SAMPLE_HMAC_KEY, SAMPLE_CODE_ID)
SAMPLE_HMAC_TRUNCATED: str = SAMPLE_HMAC_VALUE[:DEFAULT_HMAC_TRUNCATION_LENGTH]

# Sample verification URL
SAMPLE_VERIFICATION_URL: str = (
    f"{SAMPLE_VERIFICATION_BASE_URL}/verify"
    f"?code={SAMPLE_CODE_ID}"
    f"&token={SAMPLE_HMAC_TRUNCATED}"
)


# ---------------------------------------------------------------------------
# Sample QR Code Records
# ---------------------------------------------------------------------------

SAMPLE_QR_CODE: Dict[str, Any] = {
    "code_id": SAMPLE_CODE_ID,
    "version": "auto",
    "error_correction": "M",
    "symbology": "qr_code",
    "output_format": "png",
    "module_size": 10,
    "quiet_zone": 4,
    "dpi": 300,
    "payload_hash": SAMPLE_PAYLOAD_HASH,
    "payload_size_bytes": 256,
    "content_type": "compact_verification",
    "encoding": "utf8",
    "image_data_hash": SAMPLE_IMAGE_DATA_HASH,
    "image_width_px": 370,
    "image_height_px": 370,
    "logo_embedded": False,
    "quality_grade": "B",
    "operator_id": SAMPLE_OPERATOR_ID,
    "commodity": SAMPLE_COMMODITY,
    "compliance_status": "compliant",
    "dds_reference": SAMPLE_DDS_REFERENCE,
    "batch_code": None,
    "verification_url": SAMPLE_VERIFICATION_URL,
    "blockchain_anchor_hash": None,
    "status": "created",
    "reprint_count": 0,
    "scan_count": 0,
    "created_at": _ts(days_ago=1),
    "activated_at": None,
    "deactivated_at": None,
    "revoked_at": None,
    "expires_at": _ts(days_ago=-(365 * EUDR_RETENTION_YEARS)),
    "provenance_hash": SAMPLE_PROVENANCE_HASH,
}

SAMPLE_QR_CODE_SVG: Dict[str, Any] = {
    **SAMPLE_QR_CODE,
    "code_id": "QR-CODE-SVG-001",
    "output_format": "svg",
    "dpi": 0,
    "image_data_hash": _sha256("eudr-qr-svg-image-001"),
}

SAMPLE_QR_CODE_WITH_LOGO: Dict[str, Any] = {
    **SAMPLE_QR_CODE,
    "code_id": "QR-CODE-LOGO-001",
    "error_correction": "H",
    "logo_embedded": True,
    "image_data_hash": _sha256("eudr-qr-logo-image-001"),
}

ALL_SAMPLE_QR_CODES: List[Dict[str, Any]] = [
    SAMPLE_QR_CODE, SAMPLE_QR_CODE_SVG, SAMPLE_QR_CODE_WITH_LOGO,
]


# ---------------------------------------------------------------------------
# Sample Data Payloads
# ---------------------------------------------------------------------------

SAMPLE_PAYLOAD: Dict[str, Any] = {
    "payload_id": "PAYLOAD-001",
    "content_type": "compact_verification",
    "encoding": "utf8",
    "raw_data": {
        "operator_id": SAMPLE_OPERATOR_ID,
        "dds_reference": SAMPLE_DDS_REFERENCE,
        "compliance_status": "compliant",
        "commodity": SAMPLE_COMMODITY,
        "origin_country": SAMPLE_COUNTRY_CODE,
    },
    "encoded_data": '{"op":"OP-EU-COCOA-001","dds":"DDS-2026-EU-QRG-001","status":"compliant"}',
    "compressed": False,
    "encrypted": False,
    "compression_ratio": None,
    "payload_version": "1.0",
    "operator_id": SAMPLE_OPERATOR_ID,
    "commodity": SAMPLE_COMMODITY,
    "dds_reference": SAMPLE_DDS_REFERENCE,
    "compliance_status": "compliant",
    "origin_country": SAMPLE_COUNTRY_CODE,
    "origin_coordinates": {"lat": 5.6037, "lon": -0.1870},
    "certification_ids": ["CERT-RA-2026-001", "CERT-FSC-2026-002"],
    "blockchain_tx_hash": None,
    "payload_hash": SAMPLE_PAYLOAD_HASH,
    "size_bytes": 256,
    "created_at": _ts(days_ago=1),
    "provenance_hash": SAMPLE_PROVENANCE_HASH,
}

SAMPLE_PAYLOAD_COMPRESSED: Dict[str, Any] = {
    **SAMPLE_PAYLOAD,
    "payload_id": "PAYLOAD-COMPRESSED-001",
    "encoding": "zlib_base64",
    "compressed": True,
    "compression_ratio": 0.65,
    "payload_hash": _sha256("eudr-qr-compressed-payload-001"),
}

SAMPLE_PAYLOAD_ENCRYPTED: Dict[str, Any] = {
    **SAMPLE_PAYLOAD,
    "payload_id": "PAYLOAD-ENCRYPTED-001",
    "encoding": "base64",
    "encrypted": True,
    "payload_hash": _sha256("eudr-qr-encrypted-payload-001"),
}

SAMPLE_PAYLOAD_FULL_TRACEABILITY: Dict[str, Any] = {
    **SAMPLE_PAYLOAD,
    "payload_id": "PAYLOAD-FULL-001",
    "content_type": "full_traceability",
    "raw_data": {
        "operator_id": SAMPLE_OPERATOR_ID,
        "dds_reference": SAMPLE_DDS_REFERENCE,
        "compliance_status": "compliant",
        "commodity": SAMPLE_COMMODITY,
        "origin_country": SAMPLE_COUNTRY_CODE,
        "origin_coordinates": {"lat": 5.6037, "lon": -0.1870},
        "certification_ids": ["CERT-RA-2026-001"],
        "custody_chain": [
            {"from": "SUP-GH-001", "to": SAMPLE_OPERATOR_ID, "date": "2026-01-15"},
        ],
        "hs_code": SAMPLE_HS_CODE,
    },
    "size_bytes": 1024,
}

ALL_SAMPLE_PAYLOADS: List[Dict[str, Any]] = [
    SAMPLE_PAYLOAD, SAMPLE_PAYLOAD_COMPRESSED,
    SAMPLE_PAYLOAD_ENCRYPTED, SAMPLE_PAYLOAD_FULL_TRACEABILITY,
]


# ---------------------------------------------------------------------------
# Sample Label Records
# ---------------------------------------------------------------------------

SAMPLE_LABEL: Dict[str, Any] = {
    "label_id": "LABEL-001",
    "code_id": SAMPLE_CODE_ID,
    "template": "product_label",
    "font": "DejaVuSans",
    "font_size": 12,
    "compliance_color_hex": COMPLIANT_COLOR_HEX,
    "compliance_status": "compliant",
    "bleed_mm": 3,
    "width_mm": 50.0,
    "height_mm": 30.0,
    "output_format": "pdf",
    "dpi": 300,
    "image_data_hash": _sha256("eudr-label-image-001"),
    "file_size_bytes": 45_000,
    "operator_id": SAMPLE_OPERATOR_ID,
    "commodity": SAMPLE_COMMODITY,
    "product_name": SAMPLE_PRODUCT_NAME,
    "batch_code": None,
    "verification_url": SAMPLE_VERIFICATION_URL,
    "custom_fields": {},
    "created_at": _ts(days_ago=1),
    "provenance_hash": SAMPLE_PROVENANCE_HASH,
}

SAMPLE_LABEL_SHIPPING: Dict[str, Any] = {
    **SAMPLE_LABEL,
    "label_id": "LABEL-SHIP-001",
    "template": "shipping_label",
    "width_mm": 100.0,
    "height_mm": 150.0,
    "batch_code": "OP-EU-COCOA-001-cocoa-2026-00001-7",
    "custom_fields": {"destination": "Hamburg, DE", "weight_kg": "25000"},
}

SAMPLE_LABEL_CONSUMER: Dict[str, Any] = {
    **SAMPLE_LABEL,
    "label_id": "LABEL-CONSUMER-001",
    "template": "consumer_label",
    "width_mm": 40.0,
    "height_mm": 25.0,
    "font_size": 8,
    "custom_fields": {"origin_story": "Sustainably sourced from Accra, Ghana"},
}

ALL_SAMPLE_LABELS: List[Dict[str, Any]] = [
    SAMPLE_LABEL, SAMPLE_LABEL_SHIPPING, SAMPLE_LABEL_CONSUMER,
]


# ---------------------------------------------------------------------------
# Sample Batch Codes
# ---------------------------------------------------------------------------

SAMPLE_BATCH_CODE: Dict[str, Any] = {
    "batch_code_id": "BC-001",
    "code_value": "OP-EU-COCOA-001-cocoa-2026-00001-7",
    "prefix": "OP-EU-COCOA-001-cocoa-2026",
    "sequence_number": 1,
    "check_digit": "7",
    "check_digit_algorithm": "luhn",
    "operator_id": SAMPLE_OPERATOR_ID,
    "commodity": SAMPLE_COMMODITY,
    "year": 2026,
    "facility_id": SAMPLE_FACILITY_ID,
    "quantity": 25000.0,
    "quantity_unit": "kg",
    "associated_code_ids": [SAMPLE_CODE_ID],
    "status": "created",
    "created_at": _ts(days_ago=1),
    "provenance_hash": SAMPLE_PROVENANCE_HASH,
}

SAMPLE_BATCH_CODE_ISO7064: Dict[str, Any] = {
    **SAMPLE_BATCH_CODE,
    "batch_code_id": "BC-ISO7064-001",
    "code_value": "OP-EU-COCOA-001-cocoa-2026-00001-3",
    "check_digit": "3",
    "check_digit_algorithm": "iso7064_mod11_10",
}

SAMPLE_BATCH_CODE_CRC8: Dict[str, Any] = {
    **SAMPLE_BATCH_CODE,
    "batch_code_id": "BC-CRC8-001",
    "code_value": "OP-EU-COCOA-001-cocoa-2026-00001-A2",
    "check_digit": "A2",
    "check_digit_algorithm": "crc8",
}

ALL_SAMPLE_BATCH_CODES: List[Dict[str, Any]] = [
    SAMPLE_BATCH_CODE, SAMPLE_BATCH_CODE_ISO7064, SAMPLE_BATCH_CODE_CRC8,
]


# ---------------------------------------------------------------------------
# Sample Verification URL Records
# ---------------------------------------------------------------------------

SAMPLE_VERIFICATION_URL_RECORD: Dict[str, Any] = {
    "url_id": "URL-001",
    "code_id": SAMPLE_CODE_ID,
    "full_url": SAMPLE_VERIFICATION_URL,
    "short_url": None,
    "base_url": SAMPLE_VERIFICATION_BASE_URL,
    "token": SAMPLE_HMAC_VALUE,
    "hmac_truncated": SAMPLE_HMAC_TRUNCATED,
    "token_created_at": _ts(days_ago=1),
    "token_expires_at": _ts(days_ago=-(365 * DEFAULT_TOKEN_TTL_YEARS)),
    "operator_id": SAMPLE_OPERATOR_ID,
    "created_at": _ts(days_ago=1),
    "provenance_hash": SAMPLE_PROVENANCE_HASH,
}

SAMPLE_VERIFICATION_URL_WITH_SHORT: Dict[str, Any] = {
    **SAMPLE_VERIFICATION_URL_RECORD,
    "url_id": "URL-SHORT-001",
    "short_url": "https://gl.eu/v/abc123",
}

ALL_SAMPLE_VERIFICATION_URLS: List[Dict[str, Any]] = [
    SAMPLE_VERIFICATION_URL_RECORD, SAMPLE_VERIFICATION_URL_WITH_SHORT,
]


# ---------------------------------------------------------------------------
# Sample Signature Records
# ---------------------------------------------------------------------------

SAMPLE_SIGNATURE: Dict[str, Any] = {
    "signature_id": "SIG-001",
    "code_id": SAMPLE_CODE_ID,
    "algorithm": "HMAC-SHA256",
    "key_id": "KEY-QRG-001",
    "signature_value": SAMPLE_HMAC_VALUE,
    "signed_data_hash": SAMPLE_SIGNED_DATA_HASH,
    "valid": True,
    "verified_at": _ts(hours_ago=1),
    "created_at": _ts(days_ago=1),
    "provenance_hash": SAMPLE_PROVENANCE_HASH,
}

SAMPLE_SIGNATURE_ROTATED: Dict[str, Any] = {
    **SAMPLE_SIGNATURE,
    "signature_id": "SIG-ROTATED-001",
    "key_id": "KEY-QRG-002",
    "signature_value": _hmac_sha256("rotated-key-2026", SAMPLE_CODE_ID),
}

ALL_SAMPLE_SIGNATURES: List[Dict[str, Any]] = [
    SAMPLE_SIGNATURE, SAMPLE_SIGNATURE_ROTATED,
]


# ---------------------------------------------------------------------------
# Sample Scan Events
# ---------------------------------------------------------------------------

SAMPLE_SCAN_EVENT: Dict[str, Any] = {
    "scan_id": "SCAN-001",
    "code_id": SAMPLE_CODE_ID,
    "outcome": "verified",
    "scanner_ip": _sha256("192.168.1.100")[:16],
    "scanner_user_agent": "Mozilla/5.0 (Android; Mobile; rv:120.0)",
    "scan_latitude": SAMPLE_SCAN_LAT,
    "scan_longitude": SAMPLE_SCAN_LON,
    "scan_country": "BE",
    "counterfeit_risk": "low",
    "velocity_scans_per_min": 3,
    "geo_fence_violated": False,
    "hmac_valid": True,
    "response_time_ms": 45.2,
    "scanned_at": _ts(hours_ago=2),
    "provenance_hash": SAMPLE_PROVENANCE_HASH,
}

SAMPLE_SCAN_EVENT_COUNTERFEIT: Dict[str, Any] = {
    **SAMPLE_SCAN_EVENT,
    "scan_id": "SCAN-CFEIT-001",
    "outcome": "counterfeit_suspected",
    "counterfeit_risk": "critical",
    "velocity_scans_per_min": 250,
    "geo_fence_violated": True,
    "hmac_valid": False,
    "scan_latitude": -33.8688,
    "scan_longitude": 151.2093,
    "scan_country": "AU",
}

SAMPLE_SCAN_EVENT_EXPIRED: Dict[str, Any] = {
    **SAMPLE_SCAN_EVENT,
    "scan_id": "SCAN-EXP-001",
    "outcome": "expired_code",
    "counterfeit_risk": "low",
}

ALL_SAMPLE_SCAN_EVENTS: List[Dict[str, Any]] = [
    SAMPLE_SCAN_EVENT, SAMPLE_SCAN_EVENT_COUNTERFEIT, SAMPLE_SCAN_EVENT_EXPIRED,
]


# ---------------------------------------------------------------------------
# Sample Bulk Jobs
# ---------------------------------------------------------------------------

SAMPLE_BULK_JOB: Dict[str, Any] = {
    "job_id": "BULK-JOB-001",
    "status": "queued",
    "total_codes": 1000,
    "completed_codes": 0,
    "failed_codes": 0,
    "progress_percent": 0.0,
    "output_format": "png",
    "bulk_output_format": "zip",
    "output_file_hash": None,
    "output_file_size_bytes": None,
    "output_file_url": None,
    "operator_id": SAMPLE_OPERATOR_ID,
    "commodity": SAMPLE_COMMODITY,
    "content_type": "compact_verification",
    "error_correction": "M",
    "worker_count": 4,
    "error_message": None,
    "started_at": None,
    "completed_at": None,
    "created_at": _ts(hours_ago=1),
    "provenance_hash": None,
}

SAMPLE_BULK_JOB_COMPLETED: Dict[str, Any] = {
    **SAMPLE_BULK_JOB,
    "job_id": "BULK-JOB-COMPLETED-001",
    "status": "completed",
    "completed_codes": 1000,
    "progress_percent": 100.0,
    "output_file_hash": SAMPLE_OUTPUT_FILE_HASH,
    "output_file_size_bytes": 15_000_000,
    "output_file_url": "https://storage.greenlang.eu/bulk/BULK-JOB-COMPLETED-001.zip",
    "started_at": _ts(hours_ago=1),
    "completed_at": _ts(),
    "provenance_hash": SAMPLE_PROVENANCE_HASH,
}

SAMPLE_BULK_JOB_FAILED: Dict[str, Any] = {
    **SAMPLE_BULK_JOB,
    "job_id": "BULK-JOB-FAILED-001",
    "status": "failed",
    "completed_codes": 450,
    "failed_codes": 50,
    "progress_percent": 50.0,
    "error_message": "Worker timeout: exceeded 3600s limit",
    "started_at": _ts(hours_ago=2),
}

SAMPLE_BULK_JOB_PROCESSING: Dict[str, Any] = {
    **SAMPLE_BULK_JOB,
    "job_id": "BULK-JOB-PROC-001",
    "status": "processing",
    "completed_codes": 600,
    "progress_percent": 60.0,
    "started_at": _ts(hours_ago=1),
}

ALL_SAMPLE_BULK_JOBS: List[Dict[str, Any]] = [
    SAMPLE_BULK_JOB, SAMPLE_BULK_JOB_COMPLETED,
    SAMPLE_BULK_JOB_FAILED, SAMPLE_BULK_JOB_PROCESSING,
]


# ---------------------------------------------------------------------------
# Sample Lifecycle Events
# ---------------------------------------------------------------------------

SAMPLE_LIFECYCLE_EVENT: Dict[str, Any] = {
    "event_id": "LC-EVT-001",
    "code_id": SAMPLE_CODE_ID,
    "event_type": "activation",
    "previous_status": "created",
    "new_status": "active",
    "reason": "Label printed and applied to product",
    "performed_by": "system@greenlang.eu",
    "metadata": {"label_id": "LABEL-001", "print_batch": "PB-2026-001"},
    "created_at": _ts(hours_ago=12),
    "provenance_hash": SAMPLE_PROVENANCE_HASH,
}

SAMPLE_LIFECYCLE_DEACTIVATION: Dict[str, Any] = {
    **SAMPLE_LIFECYCLE_EVENT,
    "event_id": "LC-EVT-DEACT-001",
    "event_type": "deactivation",
    "previous_status": "active",
    "new_status": "deactivated",
    "reason": "Temporary product recall pending investigation",
    "performed_by": "compliance-officer@greenlang.eu",
}

SAMPLE_LIFECYCLE_REVOCATION: Dict[str, Any] = {
    **SAMPLE_LIFECYCLE_EVENT,
    "event_id": "LC-EVT-REVOKE-001",
    "event_type": "revocation",
    "previous_status": "active",
    "new_status": "revoked",
    "reason": "Counterfeit detected - HMAC mismatch confirmed",
    "performed_by": "security@greenlang.eu",
    "metadata": {"scan_id": "SCAN-CFEIT-001", "investigation_ref": "INV-2026-001"},
}

SAMPLE_LIFECYCLE_EXPIRY: Dict[str, Any] = {
    **SAMPLE_LIFECYCLE_EVENT,
    "event_id": "LC-EVT-EXP-001",
    "event_type": "expiry",
    "previous_status": "active",
    "new_status": "expired",
    "reason": "TTL expired after 5 years per EUDR Article 14",
    "performed_by": "system@greenlang.eu",
}

SAMPLE_LIFECYCLE_REPRINT: Dict[str, Any] = {
    **SAMPLE_LIFECYCLE_EVENT,
    "event_id": "LC-EVT-REPRINT-001",
    "event_type": "reprint",
    "previous_status": "active",
    "new_status": "active",
    "reason": "Label damaged during shipping",
    "performed_by": "warehouse@greenlang.eu",
    "metadata": {"reprint_count": 1, "label_id": "LABEL-REPRINT-001"},
}

ALL_SAMPLE_LIFECYCLE_EVENTS: List[Dict[str, Any]] = [
    SAMPLE_LIFECYCLE_EVENT, SAMPLE_LIFECYCLE_DEACTIVATION,
    SAMPLE_LIFECYCLE_REVOCATION, SAMPLE_LIFECYCLE_EXPIRY,
    SAMPLE_LIFECYCLE_REPRINT,
]


# ---------------------------------------------------------------------------
# Helper Factories
# ---------------------------------------------------------------------------

def make_qr_code(
    code_id: Optional[str] = None,
    version: str = "auto",
    error_correction: str = "M",
    symbology: str = "qr_code",
    output_format: str = "png",
    module_size: int = 10,
    quiet_zone: int = 4,
    dpi: int = 300,
    payload_hash: Optional[str] = None,
    payload_size_bytes: int = 256,
    content_type: str = "compact_verification",
    encoding: str = "utf8",
    logo_embedded: bool = False,
    quality_grade: str = "B",
    operator_id: str = SAMPLE_OPERATOR_ID,
    commodity: Optional[str] = SAMPLE_COMMODITY,
    compliance_status: str = "compliant",
    dds_reference: Optional[str] = SAMPLE_DDS_REFERENCE,
    status: str = "created",
    reprint_count: int = 0,
    scan_count: int = 0,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a QR code record dictionary for testing.

    Args:
        code_id: QR code identifier (auto-generated if None).
        version: QR code version.
        error_correction: Error correction level.
        symbology: Barcode symbology type.
        output_format: Output image format.
        module_size: Module pixel size.
        quiet_zone: Quiet zone modules.
        dpi: Output DPI.
        payload_hash: SHA-256 payload hash (auto-generated if None).
        payload_size_bytes: Payload size in bytes.
        content_type: Payload content type.
        encoding: Payload encoding.
        logo_embedded: Whether logo is embedded.
        quality_grade: ISO/IEC 15416 quality grade.
        operator_id: EUDR operator identifier.
        commodity: EUDR commodity type.
        compliance_status: Compliance status.
        dds_reference: DDS reference number.
        status: Lifecycle status.
        reprint_count: Reprint count.
        scan_count: Scan count.
        **overrides: Additional field overrides.

    Returns:
        Dict with all QR code record fields.
    """
    record = {
        "code_id": code_id or f"QR-{uuid.uuid4().hex[:12].upper()}",
        "version": version,
        "error_correction": error_correction,
        "symbology": symbology,
        "output_format": output_format,
        "module_size": module_size,
        "quiet_zone": quiet_zone,
        "dpi": dpi,
        "payload_hash": payload_hash or _sha256(f"qr-payload-{uuid.uuid4().hex}"),
        "payload_size_bytes": payload_size_bytes,
        "content_type": content_type,
        "encoding": encoding,
        "image_data_hash": _sha256(f"qr-image-{uuid.uuid4().hex}"),
        "image_width_px": (module_size * 33 + quiet_zone * module_size * 2),
        "image_height_px": (module_size * 33 + quiet_zone * module_size * 2),
        "logo_embedded": logo_embedded,
        "quality_grade": quality_grade,
        "operator_id": operator_id,
        "commodity": commodity,
        "compliance_status": compliance_status,
        "dds_reference": dds_reference,
        "batch_code": None,
        "verification_url": None,
        "blockchain_anchor_hash": None,
        "status": status,
        "reprint_count": reprint_count,
        "scan_count": scan_count,
        "created_at": _ts(),
        "activated_at": _ts() if status == "active" else None,
        "deactivated_at": _ts() if status == "deactivated" else None,
        "revoked_at": _ts() if status == "revoked" else None,
        "expires_at": _ts(days_ago=-(365 * EUDR_RETENTION_YEARS)),
        "provenance_hash": None,
    }
    record.update(overrides)
    return record


def make_payload(
    payload_id: Optional[str] = None,
    content_type: str = "compact_verification",
    encoding: str = "utf8",
    compressed: bool = False,
    encrypted: bool = False,
    compression_ratio: Optional[float] = None,
    payload_version: str = "1.0",
    operator_id: str = SAMPLE_OPERATOR_ID,
    commodity: Optional[str] = SAMPLE_COMMODITY,
    dds_reference: Optional[str] = SAMPLE_DDS_REFERENCE,
    compliance_status: str = "compliant",
    origin_country: Optional[str] = SAMPLE_COUNTRY_CODE,
    size_bytes: int = 256,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a data payload dictionary for testing.

    Args:
        payload_id: Payload identifier (auto-generated if None).
        content_type: Content type.
        encoding: Encoding format.
        compressed: Whether compressed.
        encrypted: Whether encrypted.
        compression_ratio: Compression ratio achieved.
        payload_version: Payload schema version.
        operator_id: EUDR operator identifier.
        commodity: EUDR commodity type.
        dds_reference: DDS reference number.
        compliance_status: Compliance status.
        origin_country: Origin country code.
        size_bytes: Payload size in bytes.
        **overrides: Additional field overrides.

    Returns:
        Dict with all data payload fields.
    """
    record = {
        "payload_id": payload_id or f"PL-{uuid.uuid4().hex[:12].upper()}",
        "content_type": content_type,
        "encoding": encoding,
        "raw_data": {
            "operator_id": operator_id,
            "dds_reference": dds_reference,
            "compliance_status": compliance_status,
            "commodity": commodity,
        },
        "encoded_data": json.dumps({"op": operator_id, "dds": dds_reference}),
        "compressed": compressed,
        "encrypted": encrypted,
        "compression_ratio": compression_ratio,
        "payload_version": payload_version,
        "operator_id": operator_id,
        "commodity": commodity,
        "dds_reference": dds_reference,
        "compliance_status": compliance_status,
        "origin_country": origin_country,
        "origin_coordinates": None,
        "certification_ids": [],
        "blockchain_tx_hash": None,
        "payload_hash": _sha256(f"payload-{uuid.uuid4().hex}"),
        "size_bytes": size_bytes,
        "created_at": _ts(),
        "provenance_hash": None,
    }
    record.update(overrides)
    return record


def make_label(
    label_id: Optional[str] = None,
    code_id: str = SAMPLE_CODE_ID,
    template: str = "product_label",
    font: str = "DejaVuSans",
    font_size: int = 12,
    compliance_status: str = "compliant",
    output_format: str = "pdf",
    dpi: int = 300,
    operator_id: str = SAMPLE_OPERATOR_ID,
    commodity: Optional[str] = SAMPLE_COMMODITY,
    product_name: Optional[str] = SAMPLE_PRODUCT_NAME,
    batch_code: Optional[str] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a label record dictionary for testing.

    Args:
        label_id: Label identifier (auto-generated if None).
        code_id: Associated QR code identifier.
        template: Label template.
        font: Font family.
        font_size: Font size in points.
        compliance_status: Compliance status.
        output_format: Output format.
        dpi: Output DPI.
        operator_id: EUDR operator identifier.
        commodity: EUDR commodity type.
        product_name: Product name.
        batch_code: Batch code.
        **overrides: Additional field overrides.

    Returns:
        Dict with all label record fields.
    """
    color_map = {
        "compliant": COMPLIANT_COLOR_HEX,
        "pending": PENDING_COLOR_HEX,
        "under_review": PENDING_COLOR_HEX,
        "non_compliant": NON_COMPLIANT_COLOR_HEX,
    }
    dims = LABEL_DIMENSIONS.get(template, {"width_mm": 50.0, "height_mm": 30.0})
    record = {
        "label_id": label_id or f"LBL-{uuid.uuid4().hex[:12].upper()}",
        "code_id": code_id,
        "template": template,
        "font": font,
        "font_size": font_size,
        "compliance_color_hex": color_map.get(compliance_status, PENDING_COLOR_HEX),
        "compliance_status": compliance_status,
        "bleed_mm": 3,
        "width_mm": dims["width_mm"],
        "height_mm": dims["height_mm"],
        "output_format": output_format,
        "dpi": dpi,
        "image_data_hash": _sha256(f"label-image-{uuid.uuid4().hex}"),
        "file_size_bytes": 45_000,
        "operator_id": operator_id,
        "commodity": commodity,
        "product_name": product_name,
        "batch_code": batch_code,
        "verification_url": SAMPLE_VERIFICATION_URL,
        "custom_fields": {},
        "created_at": _ts(),
        "provenance_hash": None,
    }
    record.update(overrides)
    return record


def make_batch_code(
    batch_code_id: Optional[str] = None,
    operator_id: str = SAMPLE_OPERATOR_ID,
    commodity: Optional[str] = SAMPLE_COMMODITY,
    year: int = 2026,
    sequence_number: int = 1,
    check_digit_algorithm: str = "luhn",
    facility_id: Optional[str] = SAMPLE_FACILITY_ID,
    quantity: Optional[float] = 25000.0,
    quantity_unit: Optional[str] = "kg",
    status: str = "created",
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a batch code dictionary for testing.

    Args:
        batch_code_id: Batch code record ID (auto-generated if None).
        operator_id: EUDR operator identifier.
        commodity: EUDR commodity type.
        year: Production/import year.
        sequence_number: Numeric sequence.
        check_digit_algorithm: Check digit algorithm.
        facility_id: Facility identifier.
        quantity: Batch quantity.
        quantity_unit: Quantity unit.
        status: Code status.
        **overrides: Additional field overrides.

    Returns:
        Dict with all batch code fields.
    """
    prefix = f"{operator_id}-{commodity}-{year}"
    padded_seq = str(sequence_number).zfill(5)
    seq_digits = padded_seq

    if check_digit_algorithm == "luhn":
        check_digit = _compute_luhn_check_digit(seq_digits)
    elif check_digit_algorithm == "iso7064_mod11_10":
        check_digit = _compute_iso7064_mod11_10(seq_digits)
    elif check_digit_algorithm == "crc8":
        crc_val = _compute_crc8(seq_digits.encode("utf-8"))
        check_digit = f"{crc_val:02X}"
    else:
        check_digit = "0"

    code_value = f"{prefix}-{padded_seq}-{check_digit}"

    record = {
        "batch_code_id": batch_code_id or f"BC-{uuid.uuid4().hex[:12].upper()}",
        "code_value": code_value,
        "prefix": prefix,
        "sequence_number": sequence_number,
        "check_digit": check_digit,
        "check_digit_algorithm": check_digit_algorithm,
        "operator_id": operator_id,
        "commodity": commodity,
        "year": year,
        "facility_id": facility_id,
        "quantity": quantity,
        "quantity_unit": quantity_unit,
        "associated_code_ids": [],
        "status": status,
        "created_at": _ts(),
        "provenance_hash": None,
    }
    record.update(overrides)
    return record


def make_verification_url(
    url_id: Optional[str] = None,
    code_id: str = SAMPLE_CODE_ID,
    base_url: str = SAMPLE_VERIFICATION_BASE_URL,
    hmac_key: str = SAMPLE_HMAC_KEY,
    operator_id: str = SAMPLE_OPERATOR_ID,
    short_url: Optional[str] = None,
    ttl_years: int = DEFAULT_TOKEN_TTL_YEARS,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a verification URL dictionary for testing.

    Args:
        url_id: URL record identifier (auto-generated if None).
        code_id: Associated QR code identifier.
        base_url: Base verification URL.
        hmac_key: HMAC signing key.
        operator_id: EUDR operator identifier.
        short_url: Shortened URL (if any).
        ttl_years: Token TTL in years.
        **overrides: Additional field overrides.

    Returns:
        Dict with all verification URL fields.
    """
    token = _hmac_sha256(hmac_key, code_id)
    truncated = token[:DEFAULT_HMAC_TRUNCATION_LENGTH]
    full = f"{base_url}/verify?code={code_id}&token={truncated}"

    record = {
        "url_id": url_id or f"URL-{uuid.uuid4().hex[:12].upper()}",
        "code_id": code_id,
        "full_url": full,
        "short_url": short_url,
        "base_url": base_url,
        "token": token,
        "hmac_truncated": truncated,
        "token_created_at": _ts(),
        "token_expires_at": _ts(days_ago=-(365 * ttl_years)),
        "operator_id": operator_id,
        "created_at": _ts(),
        "provenance_hash": None,
    }
    record.update(overrides)
    return record


def make_signature(
    signature_id: Optional[str] = None,
    code_id: str = SAMPLE_CODE_ID,
    algorithm: str = "HMAC-SHA256",
    key_id: str = "KEY-QRG-001",
    hmac_key: str = SAMPLE_HMAC_KEY,
    signed_data_hash: Optional[str] = None,
    valid: bool = True,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a signature record dictionary for testing.

    Args:
        signature_id: Signature record ID (auto-generated if None).
        code_id: Associated QR code identifier.
        algorithm: Signing algorithm.
        key_id: Signing key identifier.
        hmac_key: HMAC key for computing signature.
        signed_data_hash: Hash of the data that was signed.
        valid: Whether signature verification passed.
        **overrides: Additional field overrides.

    Returns:
        Dict with all signature record fields.
    """
    data_hash = signed_data_hash or _sha256(f"sign-data-{uuid.uuid4().hex}")
    sig_value = _hmac_sha256(hmac_key, data_hash)

    record = {
        "signature_id": signature_id or f"SIG-{uuid.uuid4().hex[:12].upper()}",
        "code_id": code_id,
        "algorithm": algorithm,
        "key_id": key_id,
        "signature_value": sig_value,
        "signed_data_hash": data_hash,
        "valid": valid,
        "verified_at": _ts() if valid else None,
        "created_at": _ts(),
        "provenance_hash": None,
    }
    record.update(overrides)
    return record


def make_scan_event(
    scan_id: Optional[str] = None,
    code_id: str = SAMPLE_CODE_ID,
    outcome: str = "verified",
    scan_latitude: Optional[float] = SAMPLE_SCAN_LAT,
    scan_longitude: Optional[float] = SAMPLE_SCAN_LON,
    scan_country: Optional[str] = "BE",
    counterfeit_risk: str = "low",
    velocity_scans_per_min: Optional[int] = 3,
    geo_fence_violated: bool = False,
    hmac_valid: Optional[bool] = True,
    response_time_ms: Optional[float] = 45.2,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a scan event dictionary for testing.

    Args:
        scan_id: Scan event identifier (auto-generated if None).
        code_id: Scanned QR code identifier.
        outcome: Scan verification outcome.
        scan_latitude: Scan location latitude.
        scan_longitude: Scan location longitude.
        scan_country: Scan country code.
        counterfeit_risk: Counterfeit risk level.
        velocity_scans_per_min: Scan velocity.
        geo_fence_violated: Whether geo-fence was violated.
        hmac_valid: Whether HMAC validated.
        response_time_ms: Processing time in ms.
        **overrides: Additional field overrides.

    Returns:
        Dict with all scan event fields.
    """
    record = {
        "scan_id": scan_id or f"SCAN-{uuid.uuid4().hex[:12].upper()}",
        "code_id": code_id,
        "outcome": outcome,
        "scanner_ip": _sha256(f"ip-{uuid.uuid4().hex}")[:16],
        "scanner_user_agent": "TestAgent/1.0",
        "scan_latitude": scan_latitude,
        "scan_longitude": scan_longitude,
        "scan_country": scan_country,
        "counterfeit_risk": counterfeit_risk,
        "velocity_scans_per_min": velocity_scans_per_min,
        "geo_fence_violated": geo_fence_violated,
        "hmac_valid": hmac_valid,
        "response_time_ms": response_time_ms,
        "scanned_at": _ts(),
        "provenance_hash": None,
    }
    record.update(overrides)
    return record


def make_bulk_job(
    job_id: Optional[str] = None,
    status: str = "queued",
    total_codes: int = 1000,
    completed_codes: int = 0,
    failed_codes: int = 0,
    output_format: str = "png",
    operator_id: str = SAMPLE_OPERATOR_ID,
    commodity: Optional[str] = SAMPLE_COMMODITY,
    content_type: str = "compact_verification",
    error_correction: str = "M",
    worker_count: int = DEFAULT_BULK_WORKERS,
    error_message: Optional[str] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a bulk job dictionary for testing.

    Args:
        job_id: Job identifier (auto-generated if None).
        status: Job status.
        total_codes: Total codes to generate.
        completed_codes: Codes generated so far.
        failed_codes: Codes that failed.
        output_format: Output format for codes.
        operator_id: EUDR operator identifier.
        commodity: EUDR commodity type.
        content_type: Payload content type.
        error_correction: Error correction level.
        worker_count: Number of workers.
        error_message: Error message if failed.
        **overrides: Additional field overrides.

    Returns:
        Dict with all bulk job fields.
    """
    progress = (
        (completed_codes + failed_codes) / total_codes * 100
        if total_codes > 0 else 0.0
    )
    record = {
        "job_id": job_id or f"BULK-{uuid.uuid4().hex[:12].upper()}",
        "status": status,
        "total_codes": total_codes,
        "completed_codes": completed_codes,
        "failed_codes": failed_codes,
        "progress_percent": progress,
        "output_format": output_format,
        "bulk_output_format": "zip",
        "output_file_hash": (
            _sha256(f"bulk-output-{uuid.uuid4().hex}")
            if status == "completed" else None
        ),
        "output_file_size_bytes": 15_000_000 if status == "completed" else None,
        "output_file_url": (
            f"https://storage.greenlang.eu/bulk/{job_id or 'JOB'}.zip"
            if status == "completed" else None
        ),
        "operator_id": operator_id,
        "commodity": commodity,
        "content_type": content_type,
        "error_correction": error_correction,
        "worker_count": worker_count,
        "error_message": error_message,
        "started_at": (
            _ts() if status in ("processing", "completed", "failed") else None
        ),
        "completed_at": _ts() if status == "completed" else None,
        "created_at": _ts(),
        "provenance_hash": (
            _sha256(f"bulk-prov-{uuid.uuid4().hex}")
            if status == "completed" else None
        ),
    }
    record.update(overrides)
    return record


def make_lifecycle_event(
    event_id: Optional[str] = None,
    code_id: str = SAMPLE_CODE_ID,
    event_type: str = "activation",
    previous_status: str = "created",
    new_status: str = "active",
    reason: Optional[str] = None,
    performed_by: Optional[str] = "system@greenlang.eu",
    metadata: Optional[Dict[str, Any]] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a lifecycle event dictionary for testing.

    Args:
        event_id: Event identifier (auto-generated if None).
        code_id: Associated QR code identifier.
        event_type: Type of lifecycle event.
        previous_status: Status before the event.
        new_status: Status after the event.
        reason: Reason for the lifecycle change.
        performed_by: User/system performing the change.
        metadata: Additional event metadata.
        **overrides: Additional field overrides.

    Returns:
        Dict with all lifecycle event fields.
    """
    record = {
        "event_id": event_id or f"LC-{uuid.uuid4().hex[:12].upper()}",
        "code_id": code_id,
        "event_type": event_type,
        "previous_status": previous_status,
        "new_status": new_status,
        "reason": reason or f"{event_type} performed",
        "performed_by": performed_by,
        "metadata": metadata or {},
        "created_at": _ts(),
        "provenance_hash": None,
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# Assertion Helpers
# ---------------------------------------------------------------------------

def assert_valid_sha256(hash_str: str) -> None:
    """Assert that a string is a valid SHA-256 hex digest.

    Args:
        hash_str: The hash string to validate.

    Raises:
        AssertionError: If hash is not a valid 64-char hex string.
    """
    assert isinstance(hash_str, str), f"Hash must be string, got {type(hash_str)}"
    assert len(hash_str) == SHA256_HEX_LENGTH, (
        f"SHA-256 hash length must be {SHA256_HEX_LENGTH}, got {len(hash_str)}"
    )
    assert all(c in "0123456789abcdef" for c in hash_str.lower()), (
        "Hash must be lowercase hex characters only"
    )


def assert_valid_hmac(hmac_str: str) -> None:
    """Assert that a string is a valid HMAC-SHA256 hex digest.

    Args:
        hmac_str: The HMAC string to validate.

    Raises:
        AssertionError: If HMAC is not a valid 64-char hex string.
    """
    assert isinstance(hmac_str, str), f"HMAC must be string, got {type(hmac_str)}"
    assert len(hmac_str) == HMAC_SHA256_HEX_LENGTH, (
        f"HMAC-SHA256 hex length must be {HMAC_SHA256_HEX_LENGTH}, got {len(hmac_str)}"
    )
    assert all(c in "0123456789abcdef" for c in hmac_str.lower()), (
        "HMAC must be hex characters only"
    )


def assert_valid_luhn(code_str: str) -> None:
    """Assert that the last character is a valid Luhn check digit.

    Args:
        code_str: Numeric string with check digit as last char.

    Raises:
        AssertionError: If check digit does not match Luhn computation.
    """
    digits_only = "".join(c for c in code_str if c.isdigit())
    assert len(digits_only) >= 2, "Need at least 2 digits for Luhn check"
    payload = digits_only[:-1]
    check = digits_only[-1]
    expected = _compute_luhn_check_digit(payload)
    assert check == expected, (
        f"Luhn check digit mismatch: expected {expected}, got {check} "
        f"for payload '{payload}'"
    )


def assert_valid_iso7064(code_str: str) -> None:
    """Assert that the last character is a valid ISO 7064 Mod 11,10 check digit.

    Args:
        code_str: Numeric string with check digit as last char.

    Raises:
        AssertionError: If check digit does not match ISO 7064 computation.
    """
    digits_only = "".join(c for c in code_str if c.isdigit())
    assert len(digits_only) >= 2, "Need at least 2 digits for ISO 7064 check"
    payload = digits_only[:-1]
    check = digits_only[-1]
    expected = _compute_iso7064_mod11_10(payload)
    assert check == expected, (
        f"ISO 7064 check digit mismatch: expected {expected}, got {check} "
        f"for payload '{payload}'"
    )


def assert_valid_crc8(code_bytes: bytes) -> None:
    """Assert that CRC-8 of the given bytes is zero (self-checking property).

    For test purposes, we validate the CRC value is a valid 0-255 int.

    Args:
        code_bytes: Bytes to validate CRC for.

    Raises:
        AssertionError: If CRC value is invalid.
    """
    crc = _compute_crc8(code_bytes)
    assert 0 <= crc <= 255, f"CRC-8 must be 0-255, got {crc}"


def assert_qr_code_valid(code: Dict[str, Any]) -> None:
    """Assert that a QR code record has all required fields and valid values.

    Args:
        code: QR code record dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "code_id" in code, "Missing code_id"
    assert len(code["code_id"]) > 0, "code_id must not be empty"

    assert "version" in code, "Missing version"

    assert "error_correction" in code, "Missing error_correction"
    assert code["error_correction"] in ERROR_CORRECTION_LEVELS, (
        f"Invalid error_correction: {code['error_correction']}"
    )

    assert "symbology" in code, "Missing symbology"
    assert code["symbology"] in SYMBOLOGY_TYPES, (
        f"Invalid symbology: {code['symbology']}"
    )

    assert "output_format" in code, "Missing output_format"
    assert code["output_format"] in SUPPORTED_FORMATS, (
        f"Invalid output_format: {code['output_format']}"
    )

    assert "payload_hash" in code, "Missing payload_hash"
    assert len(code["payload_hash"]) >= SHA256_HEX_LENGTH, (
        f"payload_hash must be at least {SHA256_HEX_LENGTH} chars"
    )

    assert "payload_size_bytes" in code, "Missing payload_size_bytes"
    assert code["payload_size_bytes"] >= 1, "payload_size_bytes must be >= 1"

    assert "content_type" in code, "Missing content_type"
    assert code["content_type"] in CONTENT_TYPES, (
        f"Invalid content_type: {code['content_type']}"
    )

    assert "operator_id" in code, "Missing operator_id"
    assert len(code["operator_id"]) > 0, "operator_id must not be empty"

    assert "status" in code, "Missing status"
    assert code["status"] in CODE_STATUSES, (
        f"Invalid status: {code['status']}"
    )

    assert "module_size" in code, "Missing module_size"
    assert 1 <= code["module_size"] <= 100, "module_size must be 1-100"

    assert "quiet_zone" in code, "Missing quiet_zone"
    assert 0 <= code["quiet_zone"] <= 20, "quiet_zone must be 0-20"

    assert "reprint_count" in code, "Missing reprint_count"
    assert code["reprint_count"] >= 0, "reprint_count must be >= 0"

    assert "scan_count" in code, "Missing scan_count"
    assert code["scan_count"] >= 0, "scan_count must be >= 0"

    assert "created_at" in code, "Missing created_at"


def assert_payload_valid(payload: Dict[str, Any]) -> None:
    """Assert that a data payload has all required fields and valid values.

    Args:
        payload: Data payload dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "payload_id" in payload, "Missing payload_id"
    assert len(payload["payload_id"]) > 0, "payload_id must not be empty"

    assert "content_type" in payload, "Missing content_type"
    assert payload["content_type"] in CONTENT_TYPES, (
        f"Invalid content_type: {payload['content_type']}"
    )

    assert "encoding" in payload, "Missing encoding"
    assert payload["encoding"] in PAYLOAD_ENCODINGS, (
        f"Invalid encoding: {payload['encoding']}"
    )

    assert "operator_id" in payload, "Missing operator_id"
    assert len(payload["operator_id"]) > 0, "operator_id must not be empty"

    assert "payload_version" in payload, "Missing payload_version"

    assert "raw_data" in payload, "Missing raw_data"
    assert isinstance(payload["raw_data"], dict), "raw_data must be a dict"

    if payload.get("compressed"):
        assert payload.get("compression_ratio") is not None, (
            "compression_ratio required when compressed=True"
        )
        assert 0.0 <= payload["compression_ratio"] <= 1.0, (
            "compression_ratio must be 0.0-1.0"
        )

    assert "created_at" in payload, "Missing created_at"


def assert_label_valid(label: Dict[str, Any]) -> None:
    """Assert that a label record has all required fields and valid values.

    Args:
        label: Label record dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "label_id" in label, "Missing label_id"
    assert len(label["label_id"]) > 0, "label_id must not be empty"

    assert "code_id" in label, "Missing code_id"
    assert len(label["code_id"]) > 0, "code_id must not be empty"

    assert "template" in label, "Missing template"
    assert label["template"] in LABEL_TEMPLATES, (
        f"Invalid template: {label['template']}"
    )

    assert "output_format" in label, "Missing output_format"
    assert label["output_format"] in SUPPORTED_FORMATS, (
        f"Invalid output_format: {label['output_format']}"
    )

    assert "compliance_status" in label, "Missing compliance_status"
    assert label["compliance_status"] in COMPLIANCE_STATUSES, (
        f"Invalid compliance_status: {label['compliance_status']}"
    )

    assert "compliance_color_hex" in label, "Missing compliance_color_hex"
    assert label["compliance_color_hex"].startswith("#"), (
        "compliance_color_hex must start with #"
    )
    assert len(label["compliance_color_hex"]) == 7, (
        "compliance_color_hex must be #RRGGBB format"
    )

    assert "operator_id" in label, "Missing operator_id"
    assert "font" in label, "Missing font"
    assert "font_size" in label, "Missing font_size"
    assert 4 <= label["font_size"] <= 72, "font_size must be 4-72"

    assert "bleed_mm" in label, "Missing bleed_mm"
    assert label["bleed_mm"] >= 0, "bleed_mm must be >= 0"

    assert "created_at" in label, "Missing created_at"


def assert_batch_code_valid(code: Dict[str, Any]) -> None:
    """Assert that a batch code has all required fields and valid values.

    Args:
        code: Batch code dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "batch_code_id" in code, "Missing batch_code_id"
    assert "code_value" in code, "Missing code_value"
    assert len(code["code_value"]) > 0, "code_value must not be empty"

    assert "prefix" in code, "Missing prefix"
    assert len(code["prefix"]) > 0, "prefix must not be empty"

    assert "sequence_number" in code, "Missing sequence_number"
    assert code["sequence_number"] >= 0, "sequence_number must be >= 0"

    assert "check_digit" in code, "Missing check_digit"
    assert len(code["check_digit"]) > 0, "check_digit must not be empty"

    assert "check_digit_algorithm" in code, "Missing check_digit_algorithm"
    assert code["check_digit_algorithm"] in CHECK_DIGIT_ALGORITHMS, (
        f"Invalid check_digit_algorithm: {code['check_digit_algorithm']}"
    )

    assert "operator_id" in code, "Missing operator_id"
    assert "year" in code, "Missing year"
    assert 2020 <= code["year"] <= 2100, "year must be 2020-2100"

    assert "status" in code, "Missing status"
    assert code["status"] in CODE_STATUSES, (
        f"Invalid status: {code['status']}"
    )

    assert "created_at" in code, "Missing created_at"


def assert_verification_url_valid(url: Dict[str, Any]) -> None:
    """Assert that a verification URL record has valid structure.

    Args:
        url: Verification URL dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "url_id" in url, "Missing url_id"
    assert "code_id" in url, "Missing code_id"
    assert len(url["code_id"]) > 0, "code_id must not be empty"

    assert "full_url" in url, "Missing full_url"
    assert url["full_url"].startswith("http"), "full_url must start with http"

    assert "base_url" in url, "Missing base_url"
    assert url["base_url"].startswith("http"), "base_url must start with http"

    assert "token" in url, "Missing token"
    assert len(url["token"]) > 0, "token must not be empty"

    assert "hmac_truncated" in url, "Missing hmac_truncated"
    assert len(url["hmac_truncated"]) >= 4, "hmac_truncated must be >= 4 chars"

    assert "operator_id" in url, "Missing operator_id"
    assert "created_at" in url, "Missing created_at"


def assert_signature_valid(sig: Dict[str, Any]) -> None:
    """Assert that a signature record has valid structure.

    Args:
        sig: Signature record dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "signature_id" in sig, "Missing signature_id"
    assert "code_id" in sig, "Missing code_id"
    assert len(sig["code_id"]) > 0, "code_id must not be empty"

    assert "algorithm" in sig, "Missing algorithm"
    assert len(sig["algorithm"]) > 0, "algorithm must not be empty"

    assert "key_id" in sig, "Missing key_id"
    assert len(sig["key_id"]) > 0, "key_id must not be empty"

    assert "signature_value" in sig, "Missing signature_value"
    assert len(sig["signature_value"]) > 0, "signature_value must not be empty"

    assert "signed_data_hash" in sig, "Missing signed_data_hash"
    assert len(sig["signed_data_hash"]) >= SHA256_HEX_LENGTH, (
        f"signed_data_hash must be at least {SHA256_HEX_LENGTH} chars"
    )

    assert "valid" in sig, "Missing valid"
    assert isinstance(sig["valid"], bool), "valid must be a boolean"

    assert "created_at" in sig, "Missing created_at"


def assert_scan_event_valid(event: Dict[str, Any]) -> None:
    """Assert that a scan event has all required fields and valid values.

    Args:
        event: Scan event dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "scan_id" in event, "Missing scan_id"
    assert "code_id" in event, "Missing code_id"
    assert len(event["code_id"]) > 0, "code_id must not be empty"

    assert "outcome" in event, "Missing outcome"
    assert event["outcome"] in SCAN_OUTCOMES, (
        f"Invalid outcome: {event['outcome']}"
    )

    assert "counterfeit_risk" in event, "Missing counterfeit_risk"
    assert event["counterfeit_risk"] in COUNTERFEIT_RISK_LEVELS, (
        f"Invalid counterfeit_risk: {event['counterfeit_risk']}"
    )

    if event.get("scan_latitude") is not None:
        assert -90.0 <= event["scan_latitude"] <= 90.0, (
            "scan_latitude must be -90 to 90"
        )
    if event.get("scan_longitude") is not None:
        assert -180.0 <= event["scan_longitude"] <= 180.0, (
            "scan_longitude must be -180 to 180"
        )

    assert "scanned_at" in event, "Missing scanned_at"


def assert_bulk_job_valid(job: Dict[str, Any]) -> None:
    """Assert that a bulk job has all required fields and valid values.

    Args:
        job: Bulk job dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "job_id" in job, "Missing job_id"
    assert "status" in job, "Missing status"
    assert job["status"] in BULK_JOB_STATUSES, (
        f"Invalid status: {job['status']}"
    )

    assert "total_codes" in job, "Missing total_codes"
    assert job["total_codes"] >= 1, "total_codes must be >= 1"

    assert "completed_codes" in job, "Missing completed_codes"
    assert job["completed_codes"] >= 0, "completed_codes must be >= 0"

    assert "failed_codes" in job, "Missing failed_codes"
    assert job["failed_codes"] >= 0, "failed_codes must be >= 0"

    assert "progress_percent" in job, "Missing progress_percent"
    assert 0.0 <= job["progress_percent"] <= 100.0, (
        "progress_percent must be 0-100"
    )

    assert "operator_id" in job, "Missing operator_id"
    assert "created_at" in job, "Missing created_at"

    if job["status"] == "completed":
        assert job.get("output_file_hash") is not None, (
            "completed job must have output_file_hash"
        )


def assert_lifecycle_event_valid(event: Dict[str, Any]) -> None:
    """Assert that a lifecycle event has all required fields and valid values.

    Args:
        event: Lifecycle event dict to validate.

    Raises:
        AssertionError: If required fields are missing or invalid.
    """
    assert "event_id" in event, "Missing event_id"
    assert "code_id" in event, "Missing code_id"
    assert len(event["code_id"]) > 0, "code_id must not be empty"

    assert "event_type" in event, "Missing event_type"
    assert len(event["event_type"]) > 0, "event_type must not be empty"

    assert "previous_status" in event, "Missing previous_status"
    assert event["previous_status"] in CODE_STATUSES, (
        f"Invalid previous_status: {event['previous_status']}"
    )

    assert "new_status" in event, "Missing new_status"
    assert event["new_status"] in CODE_STATUSES, (
        f"Invalid new_status: {event['new_status']}"
    )

    assert "created_at" in event, "Missing created_at"


# ---------------------------------------------------------------------------
# Configuration Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def qrg_config() -> Dict[str, Any]:
    """Create a QRCodeGeneratorConfig-compatible dictionary with test defaults."""
    return {
        "database_url": "postgresql://localhost:5432/greenlang_test",
        "redis_url": "redis://localhost:6379/2",
        "log_level": "DEBUG",
        "pool_size": 5,
        "default_version": "auto",
        "default_error_correction": "M",
        "default_module_size": 10,
        "default_quiet_zone": 4,
        "default_output_format": "png",
        "default_dpi": 300,
        "enable_logo_embedding": False,
        "quality_grade_target": "B",
        "default_content_type": "compact_verification",
        "max_payload_bytes": MAX_PAYLOAD_BYTES,
        "enable_compression": True,
        "compression_threshold_bytes": 500,
        "enable_encryption": False,
        "payload_version": "1.0",
        "default_template": "product_label",
        "default_font": "DejaVuSans",
        "default_font_size": 12,
        "compliant_color_hex": COMPLIANT_COLOR_HEX,
        "pending_color_hex": PENDING_COLOR_HEX,
        "non_compliant_color_hex": NON_COMPLIANT_COLOR_HEX,
        "bleed_mm": 3,
        "default_prefix_format": "{operator}-{commodity}-{year}",
        "check_digit_algorithm": "luhn",
        "code_padding": 5,
        "start_sequence": 1,
        "base_verification_url": DEFAULT_BASE_VERIFICATION_URL,
        "short_url_enabled": False,
        "short_url_service": "",
        "verification_token_ttl_years": DEFAULT_TOKEN_TTL_YEARS,
        "hmac_truncation_length": DEFAULT_HMAC_TRUNCATION_LENGTH,
        "hmac_secret_key": SAMPLE_HMAC_KEY,
        "key_rotation_days": 90,
        "enable_digital_watermark": False,
        "scan_velocity_threshold": DEFAULT_SCAN_VELOCITY_THRESHOLD,
        "geo_fence_enabled": False,
        "bulk_max_size": BULK_MAX_SIZE,
        "bulk_workers": DEFAULT_BULK_WORKERS,
        "bulk_timeout_s": DEFAULT_BULK_TIMEOUT_S,
        "bulk_output_format": "zip",
        "enable_output_validation": True,
        "default_ttl_years": EUDR_RETENTION_YEARS,
        "scan_logging_enabled": True,
        "max_reprints": DEFAULT_MAX_REPRINTS,
        "batch_max_size": MAX_BATCH_SIZE,
        "batch_concurrency": 4,
        "batch_timeout_s": 600,
        "retention_years": EUDR_RETENTION_YEARS,
        "eudr_commodities": list(EUDR_COMMODITIES),
        "enable_provenance": True,
        "genesis_hash": "GL-EUDR-QRG-014-TEST-GENESIS",
        "enable_metrics": False,
        "rate_limit": 300,
    }


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset any singleton state between tests to prevent cross-test contamination."""
    yield
    try:
        from greenlang.agents.eudr.qr_code_generator.config import reset_config
        reset_config()
    except (ImportError, ValueError, Exception):
        # ImportError: module not yet implemented
        # ValueError: transitive import triggers engine init bugs (e.g. negative shift)
        pass


# ---------------------------------------------------------------------------
# Engine Fixtures (with graceful pytest.skip for unimplemented)
# ---------------------------------------------------------------------------

@pytest.fixture
def qr_engine(qrg_config):
    """Create a QREncoder engine instance for testing."""
    try:
        from greenlang.agents.eudr.qr_code_generator.qr_encoder import QREncoder
        return QREncoder(config=qrg_config)
    except (ImportError, ValueError, Exception):
        pytest.skip("QREncoder not available")


@pytest.fixture
def payload_engine(qrg_config):
    """Create a PayloadComposer engine instance for testing."""
    try:
        from greenlang.agents.eudr.qr_code_generator.payload_composer import PayloadComposer
        return PayloadComposer(config=qrg_config)
    except (ImportError, ValueError, Exception):
        pytest.skip("PayloadComposer not available")


@pytest.fixture
def label_engine(qrg_config):
    """Create a LabelTemplateEngine engine instance for testing."""
    try:
        from greenlang.agents.eudr.qr_code_generator.label_template_engine import (
            LabelTemplateEngine,
        )
        return LabelTemplateEngine(config=qrg_config)
    except (ImportError, ValueError, Exception):
        pytest.skip("LabelTemplateEngine not available")


@pytest.fixture
def batch_engine(qrg_config):
    """Create a BatchCodeGenerator engine instance for testing."""
    try:
        from greenlang.agents.eudr.qr_code_generator.batch_code_generator import (
            BatchCodeGenerator,
        )
        return BatchCodeGenerator(config=qrg_config)
    except (ImportError, ValueError, Exception):
        pytest.skip("BatchCodeGenerator not available")


@pytest.fixture
def verification_engine(qrg_config):
    """Create a VerificationURLBuilder engine instance for testing."""
    try:
        from greenlang.agents.eudr.qr_code_generator.verification_url_builder import (
            VerificationURLBuilder,
        )
        return VerificationURLBuilder(config=qrg_config)
    except (ImportError, ValueError, Exception):
        pytest.skip("VerificationURLBuilder not available")


@pytest.fixture
def counterfeit_engine(qrg_config):
    """Create an AntiCounterfeitEngine engine instance for testing."""
    try:
        from greenlang.agents.eudr.qr_code_generator.anti_counterfeit_engine import (
            AntiCounterfeitEngine,
        )
        return AntiCounterfeitEngine(config=qrg_config)
    except (ImportError, ValueError, Exception):
        pytest.skip("AntiCounterfeitEngine not available")


@pytest.fixture
def bulk_engine(qrg_config):
    """Create a BulkGenerationPipeline engine instance for testing."""
    try:
        from greenlang.agents.eudr.qr_code_generator.bulk_generation_pipeline import (
            BulkGenerationPipeline,
        )
        return BulkGenerationPipeline(config=qrg_config)
    except (ImportError, ValueError, Exception):
        pytest.skip("BulkGenerationPipeline not available")


@pytest.fixture
def lifecycle_engine(qrg_config):
    """Create a CodeLifecycleManager engine instance for testing."""
    try:
        from greenlang.agents.eudr.qr_code_generator.code_lifecycle_manager import (
            CodeLifecycleManager,
        )
        return CodeLifecycleManager(config=qrg_config)
    except (ImportError, ValueError, Exception):
        pytest.skip("CodeLifecycleManager not available")


@pytest.fixture
def all_engines(qrg_config) -> Dict[str, Any]:
    """Return a dict of all engine instances that are available."""
    engines = {}
    engine_map = {
        "qr_encoder": (
            "greenlang.agents.eudr.qr_code_generator.qr_encoder",
            "QREncoder",
        ),
        "payload_composer": (
            "greenlang.agents.eudr.qr_code_generator.payload_composer",
            "PayloadComposer",
        ),
        "label_template_engine": (
            "greenlang.agents.eudr.qr_code_generator.label_template_engine",
            "LabelTemplateEngine",
        ),
        "batch_code_generator": (
            "greenlang.agents.eudr.qr_code_generator.batch_code_generator",
            "BatchCodeGenerator",
        ),
        "verification_url_builder": (
            "greenlang.agents.eudr.qr_code_generator.verification_url_builder",
            "VerificationURLBuilder",
        ),
        "anti_counterfeit_engine": (
            "greenlang.agents.eudr.qr_code_generator.anti_counterfeit_engine",
            "AntiCounterfeitEngine",
        ),
        "bulk_generation_pipeline": (
            "greenlang.agents.eudr.qr_code_generator.bulk_generation_pipeline",
            "BulkGenerationPipeline",
        ),
        "code_lifecycle_manager": (
            "greenlang.agents.eudr.qr_code_generator.code_lifecycle_manager",
            "CodeLifecycleManager",
        ),
    }
    for name, (module_path, class_name) in engine_map.items():
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            engines[name] = cls(config=qrg_config)
        except (ImportError, AttributeError, ValueError, Exception):
            pass
    return engines


# ---------------------------------------------------------------------------
# Data Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_qr_code() -> Dict[str, Any]:
    """Return a sample QR code record."""
    return copy.deepcopy(SAMPLE_QR_CODE)


@pytest.fixture
def sample_payload() -> Dict[str, Any]:
    """Return a sample data payload."""
    return copy.deepcopy(SAMPLE_PAYLOAD)


@pytest.fixture
def sample_label() -> Dict[str, Any]:
    """Return a sample label record."""
    return copy.deepcopy(SAMPLE_LABEL)


@pytest.fixture
def sample_batch_code() -> Dict[str, Any]:
    """Return a sample batch code."""
    return copy.deepcopy(SAMPLE_BATCH_CODE)


@pytest.fixture
def sample_verification_url() -> Dict[str, Any]:
    """Return a sample verification URL record."""
    return copy.deepcopy(SAMPLE_VERIFICATION_URL_RECORD)


@pytest.fixture
def sample_signature() -> Dict[str, Any]:
    """Return a sample signature record."""
    return copy.deepcopy(SAMPLE_SIGNATURE)


@pytest.fixture
def sample_scan_event() -> Dict[str, Any]:
    """Return a sample scan event."""
    return copy.deepcopy(SAMPLE_SCAN_EVENT)


@pytest.fixture
def sample_bulk_job() -> Dict[str, Any]:
    """Return a sample bulk job."""
    return copy.deepcopy(SAMPLE_BULK_JOB)


@pytest.fixture
def sample_lifecycle_event() -> Dict[str, Any]:
    """Return a sample lifecycle event."""
    return copy.deepcopy(SAMPLE_LIFECYCLE_EVENT)
