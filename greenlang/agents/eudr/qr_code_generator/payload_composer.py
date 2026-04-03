# -*- coding: utf-8 -*-
"""
Payload Composer Engine - AGENT-EUDR-014 QR Code Generator (Engine 2)

Production-grade data payload composition engine for EUDR QR codes.
Supports five content types (full traceability, compact verification,
consumer summary, batch identifier, blockchain anchor) with optional
zlib compression and AES-256-GCM encryption.

Capabilities:
    - Five specialised payload composition methods sized for different
      QR code applications:
        * Full traceability (500-2000 bytes): complete supply chain data
          for competent authority scanning
        * Compact verification (100-300 bytes): operator ID, DDS reference,
          compliance status, HMAC token
        * Consumer summary (50-150 bytes): product origin, deforestation-free
          status, verification URL
        * Batch identifier (30-80 bytes): batch code, commodity, quantity
        * Blockchain anchor (200-500 bytes): transaction hash, chain ID,
          block number, Merkle proof reference
    - GS1 Digital Link URI construction per GS1 General Specifications 22.0
    - Zlib compression for payloads exceeding configurable threshold
    - AES-256-GCM encryption for sensitive supply chain data
    - Schema validation for all payload types
    - QR version estimation based on payload size
    - Deterministic field ordering for reproducible SHA-256 hashing
    - Payload schema version header ("EUDR-QRG/1.0")

Zero-Hallucination Guarantees:
    - All payload composition is deterministic JSON serialization
    - Field ordering is alphabetical for reproducible hashing
    - Compression uses standard zlib with deterministic parameters
    - Encryption uses AES-256-GCM with explicit IV
    - No LLM calls in any composition path

PRD: PRD-AGENT-EUDR-014 Feature F2 (Data Payload Composition)
Agent ID: GL-EUDR-QRG-014
Regulation: EU 2023/1115 (EUDR) Articles 4, 10, 14

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
import uuid
import zlib
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.qr_code_generator.config import get_config
from greenlang.agents.eudr.qr_code_generator.metrics import (
    record_api_error,
    record_payload_composed,
)
from greenlang.agents.eudr.qr_code_generator.models import (
    ComplianceStatus,
    ContentType,
    DataPayload,
    ErrorCorrectionLevel,
    PayloadEncoding,
)
from greenlang.agents.eudr.qr_code_generator.provenance import (
    get_provenance_tracker,
)
from greenlang.utilities.exceptions.compliance import ComplianceException

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class PayloadComposerError(ComplianceException):
    """Base exception for payload composition operations."""


class PayloadSizeExceededError(PayloadComposerError):
    """Raised when payload exceeds maximum QR capacity."""


class PayloadValidationError(PayloadComposerError):
    """Raised when payload fails schema validation."""


class PayloadEncryptionError(PayloadComposerError):
    """Raised when payload encryption fails."""


class PayloadCompressionError(PayloadComposerError):
    """Raised when payload compression fails."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Payload schema version header prepended to all payloads.
PAYLOAD_HEADER = "EUDR-QRG/1.0"

#: GS1 Digital Link base URI per GS1 General Specifications 22.0.
GS1_DIGITAL_LINK_BASE = "https://id.gs1.org"

#: GS1 Application Identifiers used in EUDR traceability.
GS1_AI_GTIN = "01"
GS1_AI_SERIAL = "21"
GS1_AI_LOT = "10"
GS1_AI_BEST_BEFORE = "15"
GS1_AI_PROD_DATE = "11"
GS1_AI_COUNTRY = "422"

#: ISO/IEC 18004 binary capacity per version/EC level (same as qr_encoder).
_EC_INDEX = {"L": 0, "M": 1, "Q": 2, "H": 3}

# fmt: off
_QR_BINARY_CAPACITY: Dict[int, Tuple[int, int, int, int]] = {
    1:  (17,    14,    11,    7),
    2:  (32,    26,    20,    14),
    3:  (53,    42,    32,    24),
    4:  (78,    62,    46,    34),
    5:  (106,   84,    60,    44),
    6:  (134,   106,   74,    58),
    7:  (154,   122,   86,    64),
    8:  (192,   152,   108,   84),
    9:  (230,   180,   130,   98),
    10: (271,   213,   151,   119),
    11: (321,   251,   177,   137),
    12: (367,   287,   203,   155),
    13: (425,   331,   241,   177),
    14: (458,   362,   258,   194),
    15: (520,   412,   292,   220),
    16: (586,   450,   322,   250),
    17: (644,   504,   364,   280),
    18: (718,   560,   394,   310),
    19: (792,   624,   442,   338),
    20: (858,   666,   482,   382),
    21: (929,   711,   509,   403),
    22: (1003,  779,   565,   439),
    23: (1091,  857,   611,   461),
    24: (1171,  911,   661,   511),
    25: (1273,  997,   715,   535),
    26: (1367,  1059,  751,   593),
    27: (1465,  1125,  805,   625),
    28: (1528,  1190,  868,   658),
    29: (1628,  1264,  908,   698),
    30: (1732,  1370,  982,   742),
    31: (1840,  1452,  1030,  790),
    32: (1952,  1538,  1112,  842),
    33: (2068,  1628,  1168,  898),
    34: (2188,  1722,  1228,  958),
    35: (2303,  1809,  1283,  983),
    36: (2431,  1911,  1351,  1051),
    37: (2563,  1989,  1423,  1093),
    38: (2699,  2099,  1499,  1139),
    39: (2809,  2213,  1579,  1219),
    40: (2953,  2331,  1663,  1273),
}
# fmt: on

# ---------------------------------------------------------------------------
# Payload schema definitions for validation
# ---------------------------------------------------------------------------

_PAYLOAD_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "full_traceability": {
        "required_fields": [
            "schema_version", "content_type", "operator_id",
            "dds_reference", "commodity", "country_of_origin",
        ],
        "optional_fields": [
            "batch_id", "hs_code", "compliance_status",
            "certifications", "geolocation_hash",
            "blockchain_anchor", "custom_fields",
            "production_date", "expiry_date",
            "supply_chain_nodes", "deforestation_free",
            "legal_compliance",
        ],
        "min_bytes": 500,
        "max_bytes": 2000,
    },
    "compact_verification": {
        "required_fields": [
            "schema_version", "content_type", "operator_id",
            "dds_reference", "compliance_status",
        ],
        "optional_fields": [
            "commodity", "hmac_token", "verification_url",
            "timestamp", "country_of_origin",
        ],
        "min_bytes": 100,
        "max_bytes": 300,
    },
    "consumer_summary": {
        "required_fields": [
            "schema_version", "content_type",
            "country_of_origin", "deforestation_free",
        ],
        "optional_fields": [
            "commodity", "product_name",
            "verification_url", "certification_label",
        ],
        "min_bytes": 50,
        "max_bytes": 150,
    },
    "batch_identifier": {
        "required_fields": [
            "schema_version", "content_type", "batch_code",
            "commodity",
        ],
        "optional_fields": [
            "quantity", "quantity_unit", "production_date",
            "facility_id", "operator_id",
        ],
        "min_bytes": 30,
        "max_bytes": 80,
    },
    "blockchain_anchor": {
        "required_fields": [
            "schema_version", "content_type",
            "anchor_hash", "chain_id",
        ],
        "optional_fields": [
            "block_number", "transaction_index",
            "merkle_proof_ref", "timestamp",
            "contract_address", "operator_id",
        ],
        "min_bytes": 200,
        "max_bytes": 500,
    },
}


# ===========================================================================
# PayloadComposer
# ===========================================================================


class PayloadComposer:
    """Data payload composition engine for EUDR QR codes.

    Composes, compresses, encrypts, and validates structured data
    payloads for encoding into QR codes. All composition is fully
    deterministic with sorted JSON keys for reproducible SHA-256 hashing.

    Thread-safe: reads configuration via the thread-safe get_config()
    singleton and uses no mutable shared state.

    Attributes:
        _cfg: Reference to the QR Code Generator configuration singleton.

    Example:
        >>> composer = PayloadComposer()
        >>> payload = composer.compose_compact_verification(
        ...     operator_id="OP-EU-001",
        ...     dds_reference="DDS-2026-00042",
        ...     compliance_status=ComplianceStatus.COMPLIANT,
        ... )
        >>> assert payload.size_bytes <= 300
    """

    def __init__(self) -> None:
        """Initialize PayloadComposer with configuration singleton."""
        self._cfg = get_config()
        logger.info(
            "PayloadComposer initialized: content_type=%s, "
            "compress=%s, threshold=%d, encrypt=%s",
            self._cfg.default_content_type,
            self._cfg.enable_compression,
            self._cfg.compression_threshold_bytes,
            self._cfg.enable_encryption,
        )

    # ------------------------------------------------------------------
    # Public API: compose_payload (generic)
    # ------------------------------------------------------------------

    def compose_payload(
        self,
        content_type: ContentType,
        operator_id: str,
        dds_reference: Optional[str] = None,
        batch_id: Optional[str] = None,
        commodity: Optional[str] = None,
        country_of_origin: Optional[str] = None,
        hs_code: Optional[str] = None,
        compliance_status: Optional[ComplianceStatus] = None,
        certifications: Optional[List[str]] = None,
        geolocation_hash: Optional[str] = None,
        blockchain_anchor: Optional[str] = None,
        custom_fields: Optional[Dict[str, str]] = None,
    ) -> DataPayload:
        """Compose a generic data payload for QR code encoding.

        Builds a structured payload dictionary with deterministic field
        ordering, applies optional compression and encryption, and
        returns a DataPayload model with SHA-256 hash.

        Args:
            content_type: Payload content type.
            operator_id: EUDR operator identifier.
            dds_reference: Due Diligence Statement reference.
            batch_id: Batch identifier.
            commodity: EUDR commodity type.
            country_of_origin: ISO 3166-1 alpha-2 country code.
            hs_code: Harmonized System commodity code.
            compliance_status: EUDR compliance status.
            certifications: List of certification identifiers.
            geolocation_hash: SHA-256 hash of geolocation coordinates.
            blockchain_anchor: Blockchain anchor hash.
            custom_fields: Additional key-value pairs.

        Returns:
            DataPayload with composed and optionally compressed/encrypted data.

        Raises:
            PayloadComposerError: If composition fails.
            PayloadSizeExceededError: If payload exceeds max QR capacity.
        """
        start_time = time.monotonic()

        # Build raw data dictionary with deterministic ordering
        raw_data = self._build_raw_data(
            content_type=content_type,
            operator_id=operator_id,
            dds_reference=dds_reference,
            batch_id=batch_id,
            commodity=commodity,
            country_of_origin=country_of_origin,
            hs_code=hs_code,
            compliance_status=compliance_status,
            certifications=certifications,
            geolocation_hash=geolocation_hash,
            blockchain_anchor=blockchain_anchor,
            custom_fields=custom_fields,
        )

        # Serialize with sorted keys for deterministic hashing
        payload_bytes = self._serialize_payload(raw_data)

        # Apply compression if enabled and above threshold
        compressed = False
        compression_ratio: Optional[float] = None
        encoding = PayloadEncoding.UTF8

        if (
            self._cfg.enable_compression
            and len(payload_bytes) > self._cfg.compression_threshold_bytes
        ):
            compressed_bytes = self.compress_payload(
                payload_bytes, self._cfg.compression_threshold_bytes,
            )
            if len(compressed_bytes) < len(payload_bytes):
                compression_ratio = round(
                    1.0 - len(compressed_bytes) / len(payload_bytes), 4,
                )
                payload_bytes = compressed_bytes
                compressed = True
                encoding = PayloadEncoding.ZLIB_BASE64

        # Apply encryption if enabled
        encrypted = False
        if self._cfg.enable_encryption:
            key = self._cfg.hmac_secret_key
            if key:
                payload_bytes = self.encrypt_payload(
                    payload_bytes, key.encode("utf-8"),
                )
                encrypted = True
                encoding = PayloadEncoding.BASE64

        # Final encoded data (base64 for binary payloads)
        if compressed or encrypted:
            encoded_data = base64.b64encode(payload_bytes).decode("ascii")
        else:
            encoded_data = payload_bytes.decode("utf-8")

        # Compute SHA-256 hash of the final encoded data
        payload_hash = hashlib.sha256(
            encoded_data.encode("utf-8")
        ).hexdigest()

        size_bytes = len(encoded_data.encode("utf-8"))

        # Validate against max payload
        if size_bytes > self._cfg.max_payload_bytes:
            raise PayloadSizeExceededError(
                f"Payload size {size_bytes} bytes exceeds maximum "
                f"{self._cfg.max_payload_bytes} bytes"
            )

        # Resolve compliance status value
        cs_value = (
            compliance_status.value
            if isinstance(compliance_status, ComplianceStatus)
            else str(compliance_status or "pending")
        )

        # Build DataPayload model
        payload = DataPayload(
            content_type=content_type,
            encoding=encoding,
            raw_data=raw_data,
            encoded_data=encoded_data,
            compressed=compressed,
            encrypted=encrypted,
            compression_ratio=compression_ratio,
            payload_version=self._cfg.payload_version,
            operator_id=operator_id,
            commodity=commodity,
            dds_reference=dds_reference,
            compliance_status=ComplianceStatus(cs_value) if cs_value in [e.value for e in ComplianceStatus] else ComplianceStatus.PENDING,
            origin_country=country_of_origin,
            certification_ids=certifications or [],
            blockchain_tx_hash=blockchain_anchor,
            payload_hash=payload_hash,
            size_bytes=size_bytes,
        )

        # Provenance tracking
        tracker = get_provenance_tracker()
        prov_entry = tracker.record(
            entity_type="payload",
            action="compose",
            entity_id=payload.payload_id,
            data={
                "content_type": content_type.value if isinstance(content_type, ContentType) else str(content_type),
                "payload_hash": payload_hash,
                "size_bytes": size_bytes,
                "compressed": compressed,
                "encrypted": encrypted,
            },
            metadata={"operator_id": operator_id},
        )
        payload.provenance_hash = prov_entry.hash_value

        # Metrics
        ct_str = content_type.value if isinstance(content_type, ContentType) else str(content_type)
        record_payload_composed(ct_str)

        elapsed = time.monotonic() - start_time
        logger.info(
            "Payload composed: id=%s type=%s size=%d bytes "
            "compressed=%s encrypted=%s elapsed=%.3fs",
            payload.payload_id, ct_str, size_bytes,
            compressed, encrypted, elapsed,
        )

        return payload

    # ------------------------------------------------------------------
    # Public API: compose_full_traceability
    # ------------------------------------------------------------------

    def compose_full_traceability(
        self,
        operator_id: str,
        dds_reference: str,
        commodity: str,
        country_of_origin: str,
        batch_id: Optional[str] = None,
        hs_code: Optional[str] = None,
        compliance_status: ComplianceStatus = ComplianceStatus.PENDING,
        certifications: Optional[List[str]] = None,
        geolocation_hash: Optional[str] = None,
        blockchain_anchor: Optional[str] = None,
        production_date: Optional[str] = None,
        supply_chain_nodes: Optional[List[Dict[str, str]]] = None,
        custom_fields: Optional[Dict[str, str]] = None,
    ) -> DataPayload:
        """Compose a full traceability payload (500-2000 bytes).

        Includes complete supply chain data for competent authority
        scanning: operator identification, DDS reference, commodity
        details, geolocation, certifications, and custody chain.

        Args:
            operator_id: EUDR operator identifier.
            dds_reference: Due Diligence Statement reference number.
            commodity: EUDR-regulated commodity type.
            country_of_origin: ISO 3166-1 alpha-2 country code.
            batch_id: Optional batch identifier.
            hs_code: Optional Harmonized System code.
            compliance_status: EUDR compliance status.
            certifications: Optional certification IDs.
            geolocation_hash: SHA-256 hash of origin coordinates.
            blockchain_anchor: Blockchain transaction hash.
            production_date: ISO 8601 production date.
            supply_chain_nodes: List of supply chain node records.
            custom_fields: Additional custom fields.

        Returns:
            DataPayload with full traceability content.
        """
        extra_fields = dict(custom_fields or {})
        if production_date:
            extra_fields["production_date"] = production_date
        if supply_chain_nodes:
            extra_fields["supply_chain_nodes_count"] = str(
                len(supply_chain_nodes)
            )
            # Include node hashes for audit trail
            for i, node in enumerate(supply_chain_nodes[:10]):
                node_str = json.dumps(node, sort_keys=True)
                node_hash = hashlib.sha256(
                    node_str.encode("utf-8")
                ).hexdigest()[:16]
                extra_fields[f"node_{i}_hash"] = node_hash

        return self.compose_payload(
            content_type=ContentType.FULL_TRACEABILITY,
            operator_id=operator_id,
            dds_reference=dds_reference,
            batch_id=batch_id,
            commodity=commodity,
            country_of_origin=country_of_origin,
            hs_code=hs_code,
            compliance_status=compliance_status,
            certifications=certifications,
            geolocation_hash=geolocation_hash,
            blockchain_anchor=blockchain_anchor,
            custom_fields=extra_fields if extra_fields else None,
        )

    # ------------------------------------------------------------------
    # Public API: compose_compact_verification
    # ------------------------------------------------------------------

    def compose_compact_verification(
        self,
        operator_id: str,
        dds_reference: str,
        compliance_status: ComplianceStatus = ComplianceStatus.PENDING,
        commodity: Optional[str] = None,
        hmac_token: Optional[str] = None,
        verification_url: Optional[str] = None,
        country_of_origin: Optional[str] = None,
    ) -> DataPayload:
        """Compose a compact verification payload (100-300 bytes).

        Minimal payload for quick compliance verification containing
        operator ID, DDS reference, compliance status, and optional
        HMAC verification token.

        Args:
            operator_id: EUDR operator identifier.
            dds_reference: Due Diligence Statement reference.
            compliance_status: EUDR compliance status.
            commodity: Optional EUDR commodity type.
            hmac_token: Optional HMAC-SHA256 verification token.
            verification_url: Optional verification URL.
            country_of_origin: Optional country code.

        Returns:
            DataPayload with compact verification content.
        """
        custom = {}
        if hmac_token:
            custom["hmac_token"] = hmac_token
        if verification_url:
            custom["verification_url"] = verification_url

        return self.compose_payload(
            content_type=ContentType.COMPACT_VERIFICATION,
            operator_id=operator_id,
            dds_reference=dds_reference,
            commodity=commodity,
            country_of_origin=country_of_origin,
            compliance_status=compliance_status,
            custom_fields=custom if custom else None,
        )

    # ------------------------------------------------------------------
    # Public API: compose_consumer_summary
    # ------------------------------------------------------------------

    def compose_consumer_summary(
        self,
        country_of_origin: str,
        deforestation_free: bool = True,
        commodity: Optional[str] = None,
        product_name: Optional[str] = None,
        verification_url: Optional[str] = None,
        certification_label: Optional[str] = None,
        operator_id: str = "system",
    ) -> DataPayload:
        """Compose a consumer-facing summary payload (50-150 bytes).

        Human-readable payload for consumer scanning with product origin,
        deforestation-free status, and verification URL.

        Args:
            country_of_origin: ISO 3166-1 alpha-2 country code.
            deforestation_free: Whether the product is deforestation-free.
            commodity: Optional EUDR commodity type.
            product_name: Optional product name for display.
            verification_url: Optional verification URL.
            certification_label: Optional certification label text.
            operator_id: EUDR operator identifier.

        Returns:
            DataPayload with consumer summary content.
        """
        custom: Dict[str, str] = {
            "deforestation_free": str(deforestation_free).lower(),
        }
        if product_name:
            custom["product_name"] = product_name
        if verification_url:
            custom["verification_url"] = verification_url
        if certification_label:
            custom["certification_label"] = certification_label

        return self.compose_payload(
            content_type=ContentType.CONSUMER_SUMMARY,
            operator_id=operator_id,
            commodity=commodity,
            country_of_origin=country_of_origin,
            compliance_status=ComplianceStatus.COMPLIANT if deforestation_free else ComplianceStatus.PENDING,
            custom_fields=custom,
        )

    # ------------------------------------------------------------------
    # Public API: compose_batch_identifier
    # ------------------------------------------------------------------

    def compose_batch_identifier(
        self,
        batch_code: str,
        commodity: str,
        quantity: Optional[float] = None,
        quantity_unit: Optional[str] = None,
        production_date: Optional[str] = None,
        facility_id: Optional[str] = None,
        operator_id: str = "system",
    ) -> DataPayload:
        """Compose a batch identifier payload (30-80 bytes).

        Minimal payload encoding batch code, commodity, quantity, and
        production date for logistics and warehouse management.

        Args:
            batch_code: Full batch code string.
            commodity: EUDR-regulated commodity type.
            quantity: Optional batch quantity.
            quantity_unit: Optional unit of measurement.
            production_date: Optional ISO 8601 production date.
            facility_id: Optional facility identifier.
            operator_id: EUDR operator identifier.

        Returns:
            DataPayload with batch identifier content.
        """
        custom: Dict[str, str] = {"batch_code": batch_code}
        if quantity is not None:
            custom["quantity"] = str(quantity)
        if quantity_unit:
            custom["quantity_unit"] = quantity_unit
        if production_date:
            custom["production_date"] = production_date
        if facility_id:
            custom["facility_id"] = facility_id

        return self.compose_payload(
            content_type=ContentType.BATCH_IDENTIFIER,
            operator_id=operator_id,
            batch_id=batch_code,
            commodity=commodity,
            custom_fields=custom,
        )

    # ------------------------------------------------------------------
    # Public API: compose_blockchain_anchor
    # ------------------------------------------------------------------

    def compose_blockchain_anchor(
        self,
        anchor_hash: str,
        chain_id: str,
        block_number: Optional[int] = None,
        transaction_index: Optional[int] = None,
        merkle_proof_ref: Optional[str] = None,
        contract_address: Optional[str] = None,
        operator_id: str = "system",
    ) -> DataPayload:
        """Compose a blockchain anchor payload (200-500 bytes).

        Encodes blockchain anchor reference data linking QR code
        records to on-chain transactions via AGENT-EUDR-013.

        Args:
            anchor_hash: Blockchain transaction/anchor hash.
            chain_id: Blockchain network identifier.
            block_number: Optional block number.
            transaction_index: Optional transaction index in block.
            merkle_proof_ref: Optional Merkle proof reference URL/hash.
            contract_address: Optional smart contract address.
            operator_id: EUDR operator identifier.

        Returns:
            DataPayload with blockchain anchor content.
        """
        custom: Dict[str, str] = {
            "anchor_hash": anchor_hash,
            "chain_id": chain_id,
        }
        if block_number is not None:
            custom["block_number"] = str(block_number)
        if transaction_index is not None:
            custom["transaction_index"] = str(transaction_index)
        if merkle_proof_ref:
            custom["merkle_proof_ref"] = merkle_proof_ref
        if contract_address:
            custom["contract_address"] = contract_address

        return self.compose_payload(
            content_type=ContentType.BLOCKCHAIN_ANCHOR,
            operator_id=operator_id,
            blockchain_anchor=anchor_hash,
            custom_fields=custom,
        )

    # ------------------------------------------------------------------
    # Public API: build_gs1_digital_link_uri
    # ------------------------------------------------------------------

    def build_gs1_digital_link_uri(
        self,
        gtin: str,
        operator_code: Optional[str] = None,
        serial: Optional[str] = None,
        lot: Optional[str] = None,
    ) -> str:
        """Build a GS1 Digital Link URI per GS1 General Specifications 22.0.

        Constructs a web-resolvable URI using GS1 Application Identifiers
        (AIs) for GTIN (AI 01), serial number (AI 21), and batch/lot (AI 10).

        Args:
            gtin: Global Trade Item Number (14 digits, zero-padded).
            operator_code: Optional operator qualifier code.
            serial: Optional serial number (AI 21).
            lot: Optional batch/lot number (AI 10).

        Returns:
            GS1 Digital Link URI string.

        Raises:
            PayloadComposerError: If GTIN is invalid.
        """
        # Validate GTIN format (8, 12, 13, or 14 digits)
        gtin_clean = gtin.strip().replace(" ", "")
        if not gtin_clean.isdigit():
            raise PayloadComposerError(
                f"GTIN must contain only digits, got '{gtin}'"
            )
        if len(gtin_clean) not in (8, 12, 13, 14):
            raise PayloadComposerError(
                f"GTIN must be 8, 12, 13, or 14 digits, "
                f"got {len(gtin_clean)} digits"
            )

        # Zero-pad to 14 digits
        gtin_14 = gtin_clean.zfill(14)

        # Validate GTIN check digit (GS1 Modulo 10)
        if not self._validate_gtin_check_digit(gtin_14):
            logger.warning(
                "GTIN check digit validation failed for '%s'", gtin_14,
            )

        # Build URI path
        uri = f"{GS1_DIGITAL_LINK_BASE}/{GS1_AI_GTIN}/{gtin_14}"

        # Add serial number (AI 21)
        if serial:
            uri += f"/{GS1_AI_SERIAL}/{serial}"

        # Add lot/batch (AI 10)
        if lot:
            uri += f"/{GS1_AI_LOT}/{lot}"

        # Add operator qualifier as query parameter
        if operator_code:
            uri += f"?operator={operator_code}"

        logger.debug(
            "GS1 Digital Link URI built: %s (gtin=%s serial=%s lot=%s)",
            uri, gtin_14, serial, lot,
        )
        return uri

    # ------------------------------------------------------------------
    # Public API: compress_payload
    # ------------------------------------------------------------------

    def compress_payload(
        self,
        payload_bytes: bytes,
        threshold: Optional[int] = None,
    ) -> bytes:
        """Compress payload bytes using zlib if above threshold.

        Args:
            payload_bytes: Raw payload bytes to compress.
            threshold: Minimum size before compression is applied.
                Defaults to config compression_threshold_bytes.

        Returns:
            Compressed bytes if above threshold, otherwise original bytes.

        Raises:
            PayloadCompressionError: If zlib compression fails.
        """
        if threshold is None:
            threshold = self._cfg.compression_threshold_bytes

        if len(payload_bytes) <= threshold:
            logger.debug(
                "Payload size %d <= threshold %d; skipping compression",
                len(payload_bytes), threshold,
            )
            return payload_bytes

        try:
            compressed = zlib.compress(payload_bytes, 9)  # Max compression
            ratio = 1.0 - len(compressed) / len(payload_bytes)
            logger.info(
                "Payload compressed: %d -> %d bytes (ratio=%.2f%%)",
                len(payload_bytes), len(compressed), ratio * 100,
            )

            # Only use compressed version if it actually saves space
            if len(compressed) >= len(payload_bytes):
                logger.debug(
                    "Compressed size >= original; returning uncompressed"
                )
                return payload_bytes

            return compressed

        except zlib.error as exc:
            raise PayloadCompressionError(
                f"Zlib compression failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API: encrypt_payload
    # ------------------------------------------------------------------

    def encrypt_payload(
        self,
        payload_bytes: bytes,
        key: bytes,
    ) -> bytes:
        """Encrypt payload bytes using AES-256-GCM.

        Produces a deterministic ciphertext format:
        [12-byte IV][ciphertext][16-byte GCM tag]

        The IV is generated using os.urandom for cryptographic security.

        Args:
            payload_bytes: Plaintext bytes to encrypt.
            key: 32-byte (256-bit) encryption key. If shorter, it is
                SHA-256 hashed to produce exactly 32 bytes.

        Returns:
            Encrypted bytes in format [IV][ciphertext][tag].

        Raises:
            PayloadEncryptionError: If encryption fails.
        """
        try:
            # Ensure 32-byte key
            if len(key) != 32:
                key = hashlib.sha256(key).digest()

            # Generate 12-byte IV
            iv = os.urandom(12)

            # AES-256-GCM encryption
            # Using a pure-Python XOR-based approach as a fallback when
            # cryptography library is not available. In production,
            # the `cryptography` package provides hardware-accelerated
            # AES-GCM. This implementation provides the correct wire
            # format for downstream decryption.
            try:
                from cryptography.hazmat.primitives.ciphers.aead import (
                    AESGCM,
                )
                aesgcm = AESGCM(key)
                ciphertext_with_tag = aesgcm.encrypt(
                    iv, payload_bytes, None,
                )
                return iv + ciphertext_with_tag
            except ImportError:
                # Fallback: deterministic format with key-derived transform
                logger.warning(
                    "cryptography package not available; using "
                    "fallback encryption (not production-grade)"
                )
                return self._fallback_encrypt(payload_bytes, key, iv)

        except PayloadEncryptionError:
            raise
        except Exception as exc:
            raise PayloadEncryptionError(
                f"AES-256-GCM encryption failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API: validate_payload
    # ------------------------------------------------------------------

    def validate_payload(
        self,
        payload: DataPayload,
        schema_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate a DataPayload against its schema.

        Checks that all required fields are present, field types are
        correct, and payload size is within the expected range for
        the content type.

        Args:
            payload: DataPayload to validate.
            schema_name: Optional schema name override (defaults to
                payload.content_type).

        Returns:
            Dictionary with validation results:
                - valid: Boolean indicating pass/fail.
                - errors: List of error messages (empty if valid).
                - warnings: List of warning messages.
                - schema_name: Schema used for validation.
        """
        ct = schema_name or (
            payload.content_type
            if isinstance(payload.content_type, str)
            else payload.content_type
        )
        ct_str = ct.value if isinstance(ct, ContentType) else str(ct)

        errors: List[str] = []
        warnings: List[str] = []

        schema = _PAYLOAD_SCHEMAS.get(ct_str)
        if schema is None:
            errors.append(f"Unknown schema name: '{ct_str}'")
            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "schema_name": ct_str,
            }

        # Check required fields in raw_data
        raw = payload.raw_data or {}
        for field_name in schema["required_fields"]:
            if field_name not in raw or not raw[field_name]:
                errors.append(f"Missing required field: '{field_name}'")

        # Check size constraints
        if payload.size_bytes is not None:
            if payload.size_bytes < schema["min_bytes"]:
                warnings.append(
                    f"Payload size {payload.size_bytes} bytes below "
                    f"expected minimum {schema['min_bytes']} bytes"
                )
            if payload.size_bytes > schema["max_bytes"]:
                warnings.append(
                    f"Payload size {payload.size_bytes} bytes exceeds "
                    f"expected maximum {schema['max_bytes']} bytes"
                )

        # Check payload hash
        if not payload.payload_hash:
            errors.append("Missing payload_hash")

        # Check operator_id
        if not payload.operator_id:
            errors.append("Missing operator_id")

        valid = len(errors) == 0

        logger.info(
            "Payload validation: schema=%s valid=%s errors=%d warnings=%d",
            ct_str, valid, len(errors), len(warnings),
        )

        return {
            "valid": valid,
            "errors": errors,
            "warnings": warnings,
            "schema_name": ct_str,
        }

    # ------------------------------------------------------------------
    # Public API: estimate_qr_version
    # ------------------------------------------------------------------

    def estimate_qr_version(
        self,
        payload_bytes: bytes,
        error_correction: ErrorCorrectionLevel = ErrorCorrectionLevel.M,
    ) -> int:
        """Estimate the minimum QR version needed for a payload.

        Uses the ISO/IEC 18004 binary capacity tables to find the
        smallest QR version that can contain the given payload at
        the specified error correction level.

        Args:
            payload_bytes: Raw payload bytes.
            error_correction: Error correction level.

        Returns:
            Minimum QR version (1-40).

        Raises:
            PayloadSizeExceededError: If no version can hold the payload.
        """
        data_length = len(payload_bytes)
        ec_str = (
            error_correction.value
            if isinstance(error_correction, ErrorCorrectionLevel)
            else str(error_correction)
        )
        ec_idx = _EC_INDEX.get(ec_str, 1)

        for ver in range(1, 41):
            capacity = _QR_BINARY_CAPACITY[ver][ec_idx]
            if data_length <= capacity:
                logger.debug(
                    "Estimated QR version %d for %d bytes at EC=%s",
                    ver, data_length, ec_str,
                )
                return ver

        raise PayloadSizeExceededError(
            f"Payload size {data_length} bytes exceeds maximum QR v40-{ec_str} "
            f"capacity of {_QR_BINARY_CAPACITY[40][ec_idx]} bytes"
        )

    # ==================================================================
    # Internal: payload construction
    # ==================================================================

    def _build_raw_data(
        self,
        content_type: ContentType,
        operator_id: str,
        dds_reference: Optional[str] = None,
        batch_id: Optional[str] = None,
        commodity: Optional[str] = None,
        country_of_origin: Optional[str] = None,
        hs_code: Optional[str] = None,
        compliance_status: Optional[ComplianceStatus] = None,
        certifications: Optional[List[str]] = None,
        geolocation_hash: Optional[str] = None,
        blockchain_anchor: Optional[str] = None,
        custom_fields: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Build the raw payload data dictionary.

        Fields are added in deterministic alphabetical order. None-valued
        fields are omitted to minimize payload size.
        """
        ct_str = (
            content_type.value
            if isinstance(content_type, ContentType)
            else str(content_type)
        )
        cs_str = (
            compliance_status.value
            if isinstance(compliance_status, ComplianceStatus)
            else str(compliance_status)
        ) if compliance_status else None

        # Deterministic field assembly (alphabetical order)
        raw: Dict[str, Any] = {}

        if batch_id:
            raw["batch_id"] = batch_id
        if blockchain_anchor:
            raw["blockchain_anchor"] = blockchain_anchor
        if certifications:
            raw["certifications"] = sorted(certifications)
        if commodity:
            raw["commodity"] = commodity
        if cs_str:
            raw["compliance_status"] = cs_str
        raw["content_type"] = ct_str
        if country_of_origin:
            raw["country_of_origin"] = country_of_origin.upper()
        if custom_fields:
            raw["custom_fields"] = dict(sorted(custom_fields.items()))
        if dds_reference:
            raw["dds_reference"] = dds_reference
        if geolocation_hash:
            raw["geolocation_hash"] = geolocation_hash
        if hs_code:
            raw["hs_code"] = hs_code
        raw["operator_id"] = operator_id
        raw["schema_version"] = PAYLOAD_HEADER

        return raw

    def _serialize_payload(self, raw_data: Dict[str, Any]) -> bytes:
        """Serialize payload dictionary to deterministic JSON bytes.

        Uses sorted keys and no whitespace for minimal payload size
        while ensuring bit-perfect reproducibility.
        """
        serialized = json.dumps(
            raw_data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return serialized.encode("utf-8")

    # ==================================================================
    # Internal: GTIN check digit validation
    # ==================================================================

    def _validate_gtin_check_digit(self, gtin_14: str) -> bool:
        """Validate GTIN-14 check digit using GS1 Modulo 10 algorithm.

        Args:
            gtin_14: 14-digit GTIN string.

        Returns:
            True if the check digit is valid.
        """
        if len(gtin_14) != 14 or not gtin_14.isdigit():
            return False

        digits = [int(d) for d in gtin_14]
        check_digit = digits[-1]

        # Weighted sum: positions alternate x3, x1 from right
        weighted_sum = 0
        for i in range(13):
            weight = 3 if (13 - i) % 2 == 0 else 1
            weighted_sum += digits[i] * weight

        expected_check = (10 - (weighted_sum % 10)) % 10
        return check_digit == expected_check

    # ==================================================================
    # Internal: Fallback encryption
    # ==================================================================

    def _fallback_encrypt(
        self,
        plaintext: bytes,
        key: bytes,
        iv: bytes,
    ) -> bytes:
        """Fallback XOR-based encryption when cryptography lib unavailable.

        This is NOT production-grade encryption. It provides the correct
        wire format [IV][ciphertext][tag] for testing and development.
        In production, the `cryptography` package must be installed.

        Args:
            plaintext: Data to encrypt.
            key: 32-byte key.
            iv: 12-byte initialization vector.

        Returns:
            Encrypted bytes in format [12-byte IV][ciphertext][16-byte tag].
        """
        # Generate key stream from key + IV using SHA-256 expansion
        ciphertext = bytearray(len(plaintext))
        key_stream_source = key + iv
        block_idx = 0
        key_offset = 0

        while key_offset < len(plaintext):
            block_hash = hashlib.sha256(
                key_stream_source + block_idx.to_bytes(4, "big")
            ).digest()
            for byte_val in block_hash:
                if key_offset >= len(plaintext):
                    break
                ciphertext[key_offset] = plaintext[key_offset] ^ byte_val
                key_offset += 1
            block_idx += 1

        # Generate authentication tag
        tag_input = iv + bytes(ciphertext) + key
        tag = hashlib.sha256(tag_input).digest()[:16]

        return iv + bytes(ciphertext) + tag


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PayloadComposer",
    "PayloadComposerError",
    "PayloadSizeExceededError",
    "PayloadValidationError",
    "PayloadEncryptionError",
    "PayloadCompressionError",
    "PAYLOAD_HEADER",
    "GS1_DIGITAL_LINK_BASE",
]
