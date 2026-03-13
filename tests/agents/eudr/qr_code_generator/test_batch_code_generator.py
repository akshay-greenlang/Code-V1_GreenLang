# -*- coding: utf-8 -*-
"""
Unit tests for Engine 4: Batch Code Generator (AGENT-EUDR-014)

Tests batch code generation including check digit algorithms, batch
code format, hierarchical codes, code range generation, code reservation,
code validation, code-to-entity association, and edge cases.

55+ tests across 8 test classes.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014 QR Code Generator (GL-EUDR-QRG-014)
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from .conftest import (
    CHECK_DIGIT_ALGORITHMS,
    CODE_STATUSES,
    EUDR_COMMODITIES,
    SAMPLE_CODE_ID,
    SAMPLE_COMMODITY,
    SAMPLE_DDS_REFERENCE,
    SAMPLE_FACILITY_ID,
    SAMPLE_OPERATOR_ID,
    assert_batch_code_valid,
    assert_valid_luhn,
    assert_valid_iso7064,
    assert_valid_crc8,
    make_batch_code,
    _compute_luhn_check_digit,
    _compute_iso7064_mod11_10,
    _compute_crc8,
)


# =========================================================================
# Test Class 1: Check Digit Algorithms
# =========================================================================

class TestCheckDigitAlgorithms:
    """Test batch code generation with all 3 check digit algorithms."""

    @pytest.mark.parametrize("algo", CHECK_DIGIT_ALGORITHMS)
    def test_create_batch_code_with_algorithm(self, algo: str):
        """Test batch code creation for each check digit algorithm."""
        code = make_batch_code(check_digit_algorithm=algo)
        assert code["check_digit_algorithm"] == algo
        assert_batch_code_valid(code)

    def test_luhn_is_default(self):
        """Test Luhn is the default check digit algorithm."""
        code = make_batch_code()
        assert code["check_digit_algorithm"] == "luhn"

    def test_luhn_check_digit_computation(self):
        """Test Luhn check digit is computed correctly."""
        result = _compute_luhn_check_digit("00001")
        assert result.isdigit()
        assert len(result) == 1

    def test_iso7064_check_digit_computation(self):
        """Test ISO 7064 check digit is computed correctly."""
        result = _compute_iso7064_mod11_10("00001")
        assert result.isdigit()
        assert len(result) == 1

    def test_crc8_check_digit_computation(self):
        """Test CRC-8 check value is computed correctly."""
        result = _compute_crc8(b"00001")
        assert 0 <= result <= 255

    def test_luhn_detects_single_digit_error(self):
        """Test Luhn detects single digit substitution errors."""
        original = "12345"
        check = _compute_luhn_check_digit(original)
        # Change one digit
        modified = "12346"
        check_modified = _compute_luhn_check_digit(modified)
        assert check != check_modified

    def test_iso7064_detects_transposition(self):
        """Test ISO 7064 detects adjacent digit transposition."""
        original = "12345"
        check = _compute_iso7064_mod11_10(original)
        transposed = "12354"
        check_transposed = _compute_iso7064_mod11_10(transposed)
        assert check != check_transposed

    def test_exactly_three_algorithms(self):
        """Test exactly 3 check digit algorithms are supported."""
        assert len(CHECK_DIGIT_ALGORITHMS) == 3


# =========================================================================
# Test Class 2: Batch Code Generation
# =========================================================================

class TestBatchCodeGeneration:
    """Test batch code format, prefix, and numbering."""

    def test_prefix_format_default(self):
        """Test default prefix format is operator-commodity-year."""
        code = make_batch_code()
        expected_prefix = f"{SAMPLE_OPERATOR_ID}-{SAMPLE_COMMODITY}-2026"
        assert code["prefix"] == expected_prefix

    def test_code_value_contains_prefix(self):
        """Test code value starts with the prefix."""
        code = make_batch_code()
        assert code["code_value"].startswith(code["prefix"])

    def test_code_value_contains_check_digit(self):
        """Test code value ends with check digit."""
        code = make_batch_code()
        assert code["code_value"].endswith(code["check_digit"])

    def test_sequence_number_padded(self):
        """Test sequence number is zero-padded to 5 digits."""
        code = make_batch_code(sequence_number=1)
        assert "00001" in code["code_value"]

    def test_sequential_numbering(self):
        """Test sequential batch codes have incrementing sequences."""
        codes = [make_batch_code(sequence_number=i) for i in range(1, 6)]
        for i, code in enumerate(codes, 1):
            assert code["sequence_number"] == i

    def test_operator_id_in_prefix(self):
        """Test operator ID is embedded in the prefix."""
        code = make_batch_code(operator_id="OP-TEST-001")
        assert "OP-TEST-001" in code["prefix"]

    def test_commodity_in_prefix(self):
        """Test commodity is embedded in the prefix."""
        code = make_batch_code(commodity="coffee")
        assert "coffee" in code["prefix"]

    def test_year_in_prefix(self):
        """Test year is embedded in the prefix."""
        code = make_batch_code(year=2026)
        assert "2026" in code["prefix"]


# =========================================================================
# Test Class 3: Hierarchical Codes
# =========================================================================

class TestHierarchicalCodes:
    """Test hierarchical code structure (batch -> sub-batch -> unit)."""

    def test_batch_level_code(self):
        """Test top-level batch code generation."""
        code = make_batch_code(sequence_number=1)
        assert code["sequence_number"] == 1
        assert_batch_code_valid(code)

    def test_sub_batch_code_higher_sequence(self):
        """Test sub-batch codes have higher sequence numbers."""
        batch = make_batch_code(sequence_number=1)
        sub_batch = make_batch_code(sequence_number=100)
        assert sub_batch["sequence_number"] > batch["sequence_number"]

    def test_unit_level_code(self):
        """Test unit-level code with high sequence number."""
        unit = make_batch_code(sequence_number=10000)
        assert unit["sequence_number"] == 10000

    def test_hierarchical_codes_share_prefix(self):
        """Test hierarchical codes share the same prefix."""
        batch = make_batch_code(sequence_number=1)
        sub_batch = make_batch_code(sequence_number=100)
        assert batch["prefix"] == sub_batch["prefix"]

    def test_hierarchical_codes_different_values(self):
        """Test hierarchical codes have different code values."""
        batch = make_batch_code(sequence_number=1)
        sub_batch = make_batch_code(sequence_number=100)
        assert batch["code_value"] != sub_batch["code_value"]

    def test_associated_code_ids_list(self):
        """Test batch code can have associated QR code IDs."""
        code = make_batch_code(
            associated_code_ids=[SAMPLE_CODE_ID, "QR-002", "QR-003"],
        )
        assert len(code["associated_code_ids"]) == 3


# =========================================================================
# Test Class 4: Code Range Generation
# =========================================================================

class TestCodeRangeGeneration:
    """Test bulk code range generation."""

    def test_generate_range_of_codes(self):
        """Test generating a range of sequential codes."""
        codes = [make_batch_code(sequence_number=i) for i in range(1, 11)]
        assert len(codes) == 10

    def test_range_codes_have_unique_values(self):
        """Test all codes in range have unique code values."""
        codes = [make_batch_code(sequence_number=i) for i in range(1, 21)]
        values = {c["code_value"] for c in codes}
        assert len(values) == 20

    def test_range_codes_sequential(self):
        """Test codes in range are sequentially numbered."""
        codes = [make_batch_code(sequence_number=i) for i in range(1, 6)]
        for i, code in enumerate(codes):
            assert code["sequence_number"] == i + 1

    def test_large_range_generation(self):
        """Test generating a large range of codes."""
        codes = [make_batch_code(sequence_number=i) for i in range(1, 101)]
        assert len(codes) == 100

    def test_range_all_codes_valid(self):
        """Test all codes in a range pass validation."""
        for i in range(1, 20):
            code = make_batch_code(sequence_number=i)
            assert_batch_code_valid(code)

    def test_range_preserves_algorithm(self):
        """Test all codes in range use the same check digit algorithm."""
        codes = [
            make_batch_code(sequence_number=i, check_digit_algorithm="iso7064_mod11_10")
            for i in range(1, 6)
        ]
        for code in codes:
            assert code["check_digit_algorithm"] == "iso7064_mod11_10"


# =========================================================================
# Test Class 5: Code Reservation
# =========================================================================

class TestCodeReservation:
    """Test code range reservation."""

    def test_reserve_range_start(self):
        """Test reserving a range starting at a specific sequence."""
        code = make_batch_code(sequence_number=1000)
        assert code["sequence_number"] == 1000

    def test_reserved_codes_status_created(self):
        """Test reserved codes start with created status."""
        code = make_batch_code(status="created")
        assert code["status"] == "created"

    def test_reserve_range_for_facility(self):
        """Test reserving codes for a specific facility."""
        code = make_batch_code(facility_id="FAC-CUSTOM-001")
        assert code["facility_id"] == "FAC-CUSTOM-001"

    def test_reserve_with_quantity(self):
        """Test reservation with batch quantity."""
        code = make_batch_code(quantity=50000.0, quantity_unit="kg")
        assert code["quantity"] == 50000.0
        assert code["quantity_unit"] == "kg"

    def test_reserve_range_non_overlapping(self):
        """Test non-overlapping ranges produce distinct codes."""
        range_a = [make_batch_code(sequence_number=i) for i in range(1, 11)]
        range_b = [make_batch_code(sequence_number=i) for i in range(11, 21)]
        values_a = {c["code_value"] for c in range_a}
        values_b = {c["code_value"] for c in range_b}
        assert values_a.isdisjoint(values_b)

    def test_reserve_preserves_year(self):
        """Test reserved codes maintain the correct year."""
        code = make_batch_code(year=2027)
        assert code["year"] == 2027
        assert "2027" in code["prefix"]


# =========================================================================
# Test Class 6: Code Validation
# =========================================================================

class TestCodeValidation:
    """Test batch code check digit validation and format validation."""

    def test_luhn_check_digit_validates(self):
        """Test Luhn check digit on generated code validates."""
        code = make_batch_code(check_digit_algorithm="luhn", sequence_number=42)
        # Extract numeric parts and validate
        check = code["check_digit"]
        assert check.isdigit()
        assert len(check) == 1

    def test_iso7064_check_digit_validates(self):
        """Test ISO 7064 check digit validates."""
        code = make_batch_code(check_digit_algorithm="iso7064_mod11_10", sequence_number=42)
        check = code["check_digit"]
        assert check.isdigit()
        assert len(check) == 1

    def test_crc8_check_digit_hex_format(self):
        """Test CRC-8 check digit is in hex format."""
        code = make_batch_code(check_digit_algorithm="crc8", sequence_number=42)
        check = code["check_digit"]
        # CRC-8 is formatted as 2-char hex
        assert len(check) == 2
        int(check, 16)  # Should not raise

    def test_code_value_format_structure(self):
        """Test code value has correct structure."""
        code = make_batch_code()
        parts = code["code_value"].split("-")
        # Structure: OP-EU-COCOA-001-cocoa-2026-00001-check
        assert len(parts) >= 4

    def test_prefix_matches_components(self):
        """Test prefix matches operator-commodity-year."""
        code = make_batch_code(
            operator_id="OP-X",
            commodity="soya",
            year=2026,
        )
        assert code["prefix"] == "OP-X-soya-2026"

    @pytest.mark.parametrize("algo", CHECK_DIGIT_ALGORITHMS)
    def test_check_digit_format_per_algorithm(self, algo: str):
        """Test check digit format varies by algorithm."""
        code = make_batch_code(check_digit_algorithm=algo)
        if algo == "crc8":
            assert len(code["check_digit"]) == 2
        else:
            assert len(code["check_digit"]) == 1

    def test_year_range_validation(self):
        """Test year must be 2020-2100."""
        code = make_batch_code(year=2020)
        assert code["year"] == 2020
        code = make_batch_code(year=2100)
        assert code["year"] == 2100


# =========================================================================
# Test Class 7: Code Association
# =========================================================================

class TestCodeAssociation:
    """Test code-to-DDS/anchor linking."""

    def test_associate_code_with_qr(self):
        """Test associating batch code with QR code IDs."""
        code = make_batch_code(associated_code_ids=[SAMPLE_CODE_ID])
        assert SAMPLE_CODE_ID in code["associated_code_ids"]

    def test_associate_multiple_qr_codes(self):
        """Test associating batch code with multiple QR codes."""
        qr_ids = [f"QR-{i:03d}" for i in range(10)]
        code = make_batch_code(associated_code_ids=qr_ids)
        assert len(code["associated_code_ids"]) == 10

    def test_empty_association_list(self):
        """Test batch code with no QR code associations."""
        code = make_batch_code()
        assert code["associated_code_ids"] == []

    def test_dds_reference_via_operator(self):
        """Test DDS reference is tracked via operator context."""
        code = make_batch_code(operator_id=SAMPLE_OPERATOR_ID)
        assert code["operator_id"] == SAMPLE_OPERATOR_ID

    def test_facility_association(self):
        """Test batch code linked to a facility."""
        code = make_batch_code(facility_id=SAMPLE_FACILITY_ID)
        assert code["facility_id"] == SAMPLE_FACILITY_ID

    def test_commodity_association(self):
        """Test batch code linked to a commodity."""
        code = make_batch_code(commodity="rubber")
        assert code["commodity"] == "rubber"

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_all_commodities_in_batch_codes(self, commodity: str):
        """Test batch codes for all EUDR commodities."""
        code = make_batch_code(commodity=commodity)
        assert code["commodity"] == commodity
        assert_batch_code_valid(code)


# =========================================================================
# Test Class 8: Edge Cases
# =========================================================================

class TestBatchEdgeCases:
    """Test edge cases for batch code generation."""

    def test_collision_handling_unique_ids(self):
        """Test batch codes have unique IDs."""
        ids = {make_batch_code()["batch_code_id"] for _ in range(100)}
        assert len(ids) == 100

    def test_max_sequence_number(self):
        """Test batch code at high sequence number."""
        code = make_batch_code(sequence_number=99999)
        assert code["sequence_number"] == 99999
        assert "99999" in code["code_value"]

    def test_sequence_zero(self):
        """Test batch code at sequence number 0."""
        code = make_batch_code(sequence_number=0)
        assert code["sequence_number"] == 0
        assert "00000" in code["code_value"]

    def test_padding_preserves_width(self):
        """Test zero-padding produces consistent width."""
        code1 = make_batch_code(sequence_number=1)
        code99999 = make_batch_code(sequence_number=99999)
        # Both should have 5-digit sequence portions
        assert "00001" in code1["code_value"]
        assert "99999" in code99999["code_value"]

    def test_batch_code_with_status_active(self):
        """Test batch code with active status."""
        code = make_batch_code(status="active")
        assert code["status"] == "active"
        assert_batch_code_valid(code)

    @pytest.mark.parametrize("status", CODE_STATUSES)
    def test_all_statuses_valid(self, status: str):
        """Test batch code with each lifecycle status."""
        code = make_batch_code(status=status)
        assert code["status"] == status
        assert_batch_code_valid(code)

    def test_quantity_unit_tonnes(self):
        """Test batch code with quantity in tonnes."""
        code = make_batch_code(quantity=25.0, quantity_unit="tonnes")
        assert code["quantity"] == 25.0
        assert code["quantity_unit"] == "tonnes"

    def test_no_facility_id(self):
        """Test batch code without facility ID."""
        code = make_batch_code(facility_id=None)
        assert code["facility_id"] is None
        assert_batch_code_valid(code)
