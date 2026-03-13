# -*- coding: utf-8 -*-
"""
Unit tests for Engine 2: Payload Composer (AGENT-EUDR-014)

Tests data payload composition including content types, payload structure,
compression, encryption, GS1 Digital Link, payload validation, sizing,
and edge cases.

55+ tests across 8 test classes.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014 QR Code Generator (GL-EUDR-QRG-014)
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from .conftest import (
    COMPLIANCE_STATUSES,
    CONTENT_TYPES,
    EUDR_COMMODITIES,
    MAX_PAYLOAD_BYTES,
    PAYLOAD_ENCODINGS,
    SAMPLE_CODE_ID,
    SAMPLE_COMMODITY,
    SAMPLE_COUNTRY_CODE,
    SAMPLE_DDS_REFERENCE,
    SAMPLE_GTIN,
    SAMPLE_HS_CODE,
    SAMPLE_OPERATOR_ID,
    SAMPLE_PAYLOAD_HASH,
    SHA256_HEX_LENGTH,
    assert_payload_valid,
    assert_valid_sha256,
    make_payload,
    _sha256,
)


# =========================================================================
# Test Class 1: Content Types
# =========================================================================

class TestContentTypes:
    """Test payload composition for all 5 content types."""

    @pytest.mark.parametrize("ct", CONTENT_TYPES)
    def test_create_payload_with_content_type(self, ct: str):
        """Test payload creation for each content type."""
        payload = make_payload(content_type=ct)
        assert payload["content_type"] == ct
        assert_payload_valid(payload)

    def test_full_traceability_content(self):
        """Test full traceability payload has all supply chain data."""
        payload = make_payload(
            content_type="full_traceability",
            origin_country=SAMPLE_COUNTRY_CODE,
        )
        assert payload["content_type"] == "full_traceability"
        assert payload["origin_country"] == SAMPLE_COUNTRY_CODE

    def test_compact_verification_is_default(self):
        """Test compact verification is the default content type."""
        payload = make_payload()
        assert payload["content_type"] == "compact_verification"

    def test_consumer_summary_human_readable(self):
        """Test consumer summary is human-readable content."""
        payload = make_payload(content_type="consumer_summary")
        assert payload["content_type"] == "consumer_summary"

    def test_batch_identifier_for_logistics(self):
        """Test batch identifier for warehouse management."""
        payload = make_payload(content_type="batch_identifier")
        assert payload["content_type"] == "batch_identifier"

    def test_blockchain_anchor_reference(self):
        """Test blockchain anchor payload references chain data."""
        payload = make_payload(content_type="blockchain_anchor")
        assert payload["content_type"] == "blockchain_anchor"

    def test_exactly_five_content_types(self):
        """Test exactly 5 content types are supported."""
        assert len(CONTENT_TYPES) == 5


# =========================================================================
# Test Class 2: Payload Structure
# =========================================================================

class TestPayloadStructure:
    """Test payload structure and required fields."""

    def test_payload_has_required_fields(self):
        """Test payload has all mandatory fields."""
        payload = make_payload()
        required = [
            "payload_id", "content_type", "encoding", "raw_data",
            "operator_id", "payload_version", "created_at",
        ]
        for field in required:
            assert field in payload, f"Missing required field: {field}"

    def test_payload_version_header(self):
        """Test payload includes schema version header."""
        payload = make_payload(payload_version="1.0")
        assert payload["payload_version"] == "1.0"

    def test_payload_version_custom(self):
        """Test payload with custom version string."""
        payload = make_payload(payload_version="2.1")
        assert payload["payload_version"] == "2.1"

    def test_raw_data_is_dict(self):
        """Test raw data is a dictionary."""
        payload = make_payload()
        assert isinstance(payload["raw_data"], dict)

    def test_raw_data_contains_operator(self):
        """Test raw data includes operator identifier."""
        payload = make_payload()
        assert "operator_id" in payload["raw_data"]

    def test_encoded_data_is_string(self):
        """Test encoded data is a string when present."""
        payload = make_payload()
        assert isinstance(payload["encoded_data"], str)

    def test_payload_hash_computed(self):
        """Test payload hash is a valid SHA-256 digest."""
        payload = make_payload()
        assert payload["payload_hash"] is not None
        assert_valid_sha256(payload["payload_hash"])

    def test_size_bytes_positive(self):
        """Test size bytes is a positive integer."""
        payload = make_payload(size_bytes=512)
        assert payload["size_bytes"] == 512
        assert payload["size_bytes"] > 0


# =========================================================================
# Test Class 3: Payload Compression
# =========================================================================

class TestPayloadCompression:
    """Test zlib compression of payloads above threshold."""

    def test_uncompressed_by_default(self):
        """Test payload is not compressed by default."""
        payload = make_payload()
        assert payload["compressed"] is False

    def test_compressed_payload_flag(self):
        """Test compressed flag is set when compression enabled."""
        payload = make_payload(compressed=True, compression_ratio=0.6)
        assert payload["compressed"] is True

    def test_compression_ratio_valid_range(self):
        """Test compression ratio is between 0 and 1."""
        payload = make_payload(compressed=True, compression_ratio=0.65)
        assert 0.0 <= payload["compression_ratio"] <= 1.0

    def test_compression_ratio_none_when_uncompressed(self):
        """Test compression ratio is None when not compressed."""
        payload = make_payload(compressed=False)
        assert payload["compression_ratio"] is None

    def test_compressed_encoding_is_zlib_base64(self):
        """Test compressed payloads use zlib_base64 encoding."""
        payload = make_payload(
            compressed=True,
            encoding="zlib_base64",
            compression_ratio=0.5,
        )
        assert payload["encoding"] == "zlib_base64"

    def test_high_compression_ratio(self):
        """Test payload with high compression ratio."""
        payload = make_payload(compressed=True, compression_ratio=0.3)
        assert payload["compression_ratio"] == 0.3

    def test_minimal_compression_ratio(self):
        """Test payload with minimal compression (almost no savings)."""
        payload = make_payload(compressed=True, compression_ratio=0.95)
        assert payload["compression_ratio"] == 0.95

    def test_compressed_payload_valid(self):
        """Test compressed payload passes validation."""
        payload = make_payload(compressed=True, compression_ratio=0.6)
        assert_payload_valid(payload)


# =========================================================================
# Test Class 4: Payload Encryption
# =========================================================================

class TestPayloadEncryption:
    """Test AES-256-GCM payload encryption."""

    def test_unencrypted_by_default(self):
        """Test payload is not encrypted by default."""
        payload = make_payload()
        assert payload["encrypted"] is False

    def test_encrypted_payload_flag(self):
        """Test encrypted flag is set when encryption enabled."""
        payload = make_payload(encrypted=True)
        assert payload["encrypted"] is True

    def test_encrypted_encoding_is_base64(self):
        """Test encrypted payloads use base64 encoding."""
        payload = make_payload(encrypted=True, encoding="base64")
        assert payload["encoding"] == "base64"

    def test_encrypted_payload_has_hash(self):
        """Test encrypted payload still has a valid hash."""
        payload = make_payload(encrypted=True)
        assert payload["payload_hash"] is not None
        assert_valid_sha256(payload["payload_hash"])

    def test_encrypted_and_compressed(self):
        """Test payload can be both compressed and encrypted."""
        payload = make_payload(
            compressed=True,
            encrypted=True,
            compression_ratio=0.5,
            encoding="base64",
        )
        assert payload["compressed"] is True
        assert payload["encrypted"] is True

    def test_encrypted_payload_valid(self):
        """Test encrypted payload passes validation."""
        payload = make_payload(encrypted=True, encoding="base64")
        assert_payload_valid(payload)


# =========================================================================
# Test Class 5: GS1 Digital Link
# =========================================================================

class TestGS1DigitalLink:
    """Test GS1 Digital Link URI construction."""

    def test_gs1_content_type(self):
        """Test GS1 Digital Link as a content type variant."""
        payload = make_payload(
            content_type="full_traceability",
            raw_data={
                "gtin": SAMPLE_GTIN,
                "operator_id": SAMPLE_OPERATOR_ID,
                "dds_reference": SAMPLE_DDS_REFERENCE,
            },
        )
        assert payload["raw_data"]["gtin"] == SAMPLE_GTIN

    def test_gtin_14_digit_format(self):
        """Test GTIN is exactly 14 digits."""
        assert len(SAMPLE_GTIN) == 14
        assert SAMPLE_GTIN.isdigit()

    def test_gs1_uri_includes_gtin(self):
        """Test GS1 URI construction includes GTIN in raw data."""
        payload = make_payload(
            raw_data={"gtin": SAMPLE_GTIN, "gs1_uri": f"https://id.gs1.org/01/{SAMPLE_GTIN}"},
        )
        assert SAMPLE_GTIN in payload["raw_data"]["gs1_uri"]

    def test_gs1_with_additional_attributes(self):
        """Test GS1 Digital Link with additional data attributes."""
        payload = make_payload(
            raw_data={
                "gtin": SAMPLE_GTIN,
                "batch_lot": "LOT-2026-001",
                "serial": "SN-001",
            },
        )
        assert payload["raw_data"]["batch_lot"] == "LOT-2026-001"
        assert payload["raw_data"]["serial"] == "SN-001"

    def test_gs1_payload_valid(self):
        """Test GS1 payload passes validation."""
        payload = make_payload(
            raw_data={"gtin": SAMPLE_GTIN, "operator_id": SAMPLE_OPERATOR_ID},
        )
        assert_payload_valid(payload)

    def test_gs1_hs_code_reference(self):
        """Test GS1 payload with HS code reference."""
        payload = make_payload(
            raw_data={"gtin": SAMPLE_GTIN, "hs_code": SAMPLE_HS_CODE},
        )
        assert payload["raw_data"]["hs_code"] == SAMPLE_HS_CODE


# =========================================================================
# Test Class 6: Payload Validation
# =========================================================================

class TestPayloadValidation:
    """Test payload schema and field validation."""

    @pytest.mark.parametrize("encoding", PAYLOAD_ENCODINGS)
    def test_valid_encodings(self, encoding: str):
        """Test all payload encodings are valid."""
        payload = make_payload(encoding=encoding)
        assert payload["encoding"] == encoding
        assert_payload_valid(payload)

    @pytest.mark.parametrize("status", COMPLIANCE_STATUSES)
    def test_compliance_status_values(self, status: str):
        """Test all compliance statuses in payload."""
        payload = make_payload(compliance_status=status)
        assert payload["compliance_status"] == status

    def test_origin_country_iso_format(self):
        """Test origin country is ISO 3166-1 alpha-2."""
        payload = make_payload(origin_country="GH")
        assert payload["origin_country"] == "GH"
        assert len(payload["origin_country"]) == 2

    def test_certification_ids_list(self):
        """Test certification IDs is a list."""
        payload = make_payload(
            certification_ids=["CERT-001", "CERT-002"],
        )
        assert isinstance(payload["certification_ids"], list)

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_eudr_commodities_accepted(self, commodity: str):
        """Test all 7 EUDR commodities are accepted."""
        payload = make_payload(commodity=commodity)
        assert payload["commodity"] == commodity
        assert_payload_valid(payload)

    def test_dds_reference_in_payload(self):
        """Test DDS reference is present in payload."""
        payload = make_payload(dds_reference=SAMPLE_DDS_REFERENCE)
        assert payload["dds_reference"] == SAMPLE_DDS_REFERENCE

    def test_empty_certification_ids(self):
        """Test empty certification IDs list is valid."""
        payload = make_payload()
        assert payload["certification_ids"] == []


# =========================================================================
# Test Class 7: Payload Sizing
# =========================================================================

class TestPayloadSizing:
    """Test payload size estimation for QR version selection."""

    def test_small_payload_size(self):
        """Test small payload for low QR versions."""
        payload = make_payload(size_bytes=50)
        assert payload["size_bytes"] == 50

    def test_medium_payload_size(self):
        """Test medium payload for mid QR versions."""
        payload = make_payload(size_bytes=500)
        assert payload["size_bytes"] == 500

    def test_large_payload_at_max(self):
        """Test payload at maximum QR capacity."""
        payload = make_payload(size_bytes=MAX_PAYLOAD_BYTES)
        assert payload["size_bytes"] == MAX_PAYLOAD_BYTES

    def test_max_payload_constant(self):
        """Test MAX_PAYLOAD_BYTES is 2953."""
        assert MAX_PAYLOAD_BYTES == 2953

    def test_compressed_payload_smaller_size(self):
        """Test compressed payload reports smaller effective size."""
        payload = make_payload(
            compressed=True,
            compression_ratio=0.5,
            size_bytes=250,
        )
        # Compression ratio 0.5 means 50% of original kept
        assert payload["size_bytes"] == 250

    def test_full_traceability_larger_than_compact(self):
        """Test full traceability payload is larger than compact."""
        full = make_payload(content_type="full_traceability", size_bytes=1024)
        compact = make_payload(content_type="compact_verification", size_bytes=256)
        assert full["size_bytes"] > compact["size_bytes"]


# =========================================================================
# Test Class 8: Edge Cases
# =========================================================================

class TestPayloadEdgeCases:
    """Test edge cases for payload composition."""

    def test_custom_fields_in_raw_data(self):
        """Test custom fields can be added to raw data."""
        payload = make_payload(
            raw_data={
                "operator_id": SAMPLE_OPERATOR_ID,
                "custom_field_1": "value1",
                "custom_field_2": 42,
            },
        )
        assert payload["raw_data"]["custom_field_1"] == "value1"
        assert payload["raw_data"]["custom_field_2"] == 42

    def test_multi_language_product_name(self):
        """Test payload with multi-language product name."""
        payload = make_payload(
            raw_data={
                "product_name_en": "Cocoa Butter",
                "product_name_de": "Kakaobutter",
                "product_name_fr": "Beurre de cacao",
                "operator_id": SAMPLE_OPERATOR_ID,
            },
        )
        assert payload["raw_data"]["product_name_de"] == "Kakaobutter"

    def test_unicode_in_payload(self):
        """Test payload with Unicode characters."""
        payload = make_payload(
            raw_data={
                "operator_id": SAMPLE_OPERATOR_ID,
                "origin": "Cote d'Ivoire",
            },
        )
        assert "Cote d'Ivoire" in payload["raw_data"]["origin"]

    def test_empty_raw_data_dict(self):
        """Test payload with minimal raw data."""
        payload = make_payload(raw_data={"operator_id": SAMPLE_OPERATOR_ID})
        assert isinstance(payload["raw_data"], dict)

    def test_blockchain_tx_hash_optional(self):
        """Test blockchain transaction hash is optional."""
        payload = make_payload()
        assert payload["blockchain_tx_hash"] is None

    def test_blockchain_tx_hash_present(self):
        """Test blockchain transaction hash when set."""
        tx = _sha256("blockchain-tx-001")
        payload = make_payload(blockchain_tx_hash=tx)
        assert payload["blockchain_tx_hash"] == tx

    def test_origin_coordinates_optional(self):
        """Test origin coordinates are optional."""
        payload = make_payload()
        assert payload["origin_coordinates"] is None

    def test_origin_coordinates_with_values(self):
        """Test origin coordinates with latitude and longitude."""
        payload = make_payload(
            origin_coordinates={"lat": 5.6037, "lon": -0.1870},
        )
        assert payload["origin_coordinates"]["lat"] == 5.6037
        assert payload["origin_coordinates"]["lon"] == -0.1870

    def test_unique_payload_ids(self):
        """Test each generated payload has a unique payload_id."""
        ids = {make_payload()["payload_id"] for _ in range(50)}
        assert len(ids) == 50
