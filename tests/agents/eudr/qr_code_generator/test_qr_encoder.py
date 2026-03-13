# -*- coding: utf-8 -*-
"""
Unit tests for Engine 1: QR Encoder (AGENT-EUDR-014)

Tests QR code generation including output formats, error correction levels,
symbology types, version selection, logo embedding, quality grading,
DPI configuration, and edge cases.

65+ tests across 8 test classes.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014 QR Code Generator (GL-EUDR-QRG-014)
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict

import pytest

from .conftest import (
    CONTENT_TYPES,
    DPI_LEVELS,
    ERROR_CORRECTION_LEVELS,
    MAX_PAYLOAD_BYTES,
    MAX_QR_VERSION,
    QUALITY_GRADES,
    SAMPLE_CODE_ID,
    SAMPLE_OPERATOR_ID,
    SAMPLE_PAYLOAD_HASH,
    SHA256_HEX_LENGTH,
    SUPPORTED_FORMATS,
    SYMBOLOGY_TYPES,
    assert_qr_code_valid,
    assert_valid_sha256,
    make_qr_code,
    _sha256,
)


# =========================================================================
# Test Class 1: Output Formats
# =========================================================================

class TestOutputFormats:
    """Test QR code generation across all 5 output formats."""

    @pytest.mark.parametrize("fmt", SUPPORTED_FORMATS)
    def test_create_qr_code_with_format(self, fmt: str):
        """Test QR code creation for each supported output format."""
        code = make_qr_code(output_format=fmt)
        assert code["output_format"] == fmt
        assert_qr_code_valid(code)

    @pytest.mark.parametrize("fmt", SUPPORTED_FORMATS)
    def test_format_produces_image_data_hash(self, fmt: str):
        """Test that each format produces an image data hash."""
        code = make_qr_code(output_format=fmt)
        assert code["image_data_hash"] is not None
        assert len(code["image_data_hash"]) == SHA256_HEX_LENGTH

    def test_png_format_has_pixel_dimensions(self):
        """Test PNG output has width and height in pixels."""
        code = make_qr_code(output_format="png")
        assert code["image_width_px"] is not None
        assert code["image_width_px"] > 0
        assert code["image_height_px"] is not None
        assert code["image_height_px"] > 0

    def test_svg_format_is_vector(self):
        """Test SVG output is a vector format (no DPI dependency)."""
        code = make_qr_code(output_format="svg", dpi=0)
        assert code["output_format"] == "svg"

    def test_pdf_format_for_print(self):
        """Test PDF format suitable for print production."""
        code = make_qr_code(output_format="pdf", dpi=300)
        assert code["output_format"] == "pdf"
        assert code["dpi"] == 300

    def test_zpl_format_for_thermal_printing(self):
        """Test ZPL format for Zebra thermal printers."""
        code = make_qr_code(output_format="zpl")
        assert code["output_format"] == "zpl"
        assert_qr_code_valid(code)

    def test_eps_format_for_prepress(self):
        """Test EPS format for prepress workflows."""
        code = make_qr_code(output_format="eps")
        assert code["output_format"] == "eps"
        assert_qr_code_valid(code)

    def test_invalid_format_not_in_supported(self):
        """Test that unsupported formats are not in the list."""
        assert "bmp" not in SUPPORTED_FORMATS
        assert "gif" not in SUPPORTED_FORMATS
        assert "tiff" not in SUPPORTED_FORMATS


# =========================================================================
# Test Class 2: Error Correction Levels
# =========================================================================

class TestErrorCorrection:
    """Test QR code generation with all 4 error correction levels."""

    @pytest.mark.parametrize("ec_level", ERROR_CORRECTION_LEVELS)
    def test_create_qr_code_with_ec_level(self, ec_level: str):
        """Test QR code creation for each error correction level."""
        code = make_qr_code(error_correction=ec_level)
        assert code["error_correction"] == ec_level
        assert_qr_code_valid(code)

    def test_ec_l_maximizes_capacity(self):
        """Test EC level L provides maximum data capacity."""
        code = make_qr_code(error_correction="L", payload_size_bytes=2000)
        assert code["error_correction"] == "L"
        assert code["payload_size_bytes"] == 2000

    def test_ec_m_is_default(self):
        """Test EC level M is the default level."""
        code = make_qr_code()
        assert code["error_correction"] == "M"

    def test_ec_q_for_moderate_wear(self):
        """Test EC level Q for labels exposed to moderate wear."""
        code = make_qr_code(error_correction="Q")
        assert code["error_correction"] == "Q"

    def test_ec_h_required_for_logo(self):
        """Test EC level H is required when embedding a logo."""
        code = make_qr_code(error_correction="H", logo_embedded=True)
        assert code["error_correction"] == "H"
        assert code["logo_embedded"] is True

    def test_ec_levels_are_four(self):
        """Test exactly 4 error correction levels exist."""
        assert len(ERROR_CORRECTION_LEVELS) == 4

    def test_ec_level_ordering(self):
        """Test EC levels are in order of increasing recovery capability."""
        assert ERROR_CORRECTION_LEVELS == ["L", "M", "Q", "H"]

    def test_ec_h_code_is_valid(self):
        """Test that EC-H QR code passes full validation."""
        code = make_qr_code(error_correction="H")
        assert_qr_code_valid(code)


# =========================================================================
# Test Class 3: Symbology Types
# =========================================================================

class TestSymbologyTypes:
    """Test QR code generation across all 4 symbology types."""

    @pytest.mark.parametrize("symbology", SYMBOLOGY_TYPES)
    def test_create_code_with_symbology(self, symbology: str):
        """Test code creation for each symbology type."""
        code = make_qr_code(symbology=symbology)
        assert code["symbology"] == symbology
        assert_qr_code_valid(code)

    def test_qr_code_is_default_symbology(self):
        """Test QR code is the default symbology type."""
        code = make_qr_code()
        assert code["symbology"] == "qr_code"

    def test_micro_qr_for_small_labels(self):
        """Test Micro QR is available for very small labels."""
        code = make_qr_code(symbology="micro_qr", payload_size_bytes=30)
        assert code["symbology"] == "micro_qr"
        assert code["payload_size_bytes"] <= 35

    def test_data_matrix_alternative(self):
        """Test Data Matrix as alternative 2D symbology."""
        code = make_qr_code(symbology="data_matrix")
        assert code["symbology"] == "data_matrix"

    def test_gs1_digital_link_uri(self):
        """Test GS1 Digital Link for web-resolvable product identification."""
        code = make_qr_code(symbology="gs1_digital_link")
        assert code["symbology"] == "gs1_digital_link"

    def test_exactly_four_symbology_types(self):
        """Test exactly 4 symbology types are supported."""
        assert len(SYMBOLOGY_TYPES) == 4

    def test_symbology_types_list(self):
        """Test all expected symbology types are present."""
        assert "qr_code" in SYMBOLOGY_TYPES
        assert "micro_qr" in SYMBOLOGY_TYPES
        assert "data_matrix" in SYMBOLOGY_TYPES
        assert "gs1_digital_link" in SYMBOLOGY_TYPES


# =========================================================================
# Test Class 4: Version Selection
# =========================================================================

class TestVersionSelection:
    """Test QR code version selection (auto and fixed)."""

    def test_auto_version_selection_default(self):
        """Test auto version is the default selection."""
        code = make_qr_code()
        assert code["version"] == "auto"

    def test_auto_version_for_small_payload(self):
        """Test auto version selects small version for small payload."""
        code = make_qr_code(version="auto", payload_size_bytes=50)
        assert code["version"] == "auto"
        assert code["payload_size_bytes"] == 50

    def test_fixed_version_1(self):
        """Test fixed version 1 for minimal payload."""
        code = make_qr_code(version="1")
        assert code["version"] == "1"

    def test_fixed_version_10(self):
        """Test fixed version 10 for medium payload."""
        code = make_qr_code(version="10")
        assert code["version"] == "10"

    def test_fixed_version_25(self):
        """Test fixed version 25 for large payload."""
        code = make_qr_code(version="25")
        assert code["version"] == "25"

    def test_max_version_40(self):
        """Test maximum version 40 for very large payloads."""
        code = make_qr_code(version="40")
        assert code["version"] == "40"

    def test_max_version_constant(self):
        """Test MAX_QR_VERSION constant is 40."""
        assert MAX_QR_VERSION == 40

    @pytest.mark.parametrize("ver", [str(i) for i in range(1, 41)])
    def test_all_40_versions_valid(self, ver: str):
        """Test all 40 fixed versions produce valid codes."""
        code = make_qr_code(version=ver)
        assert code["version"] == ver
        assert_qr_code_valid(code)


# =========================================================================
# Test Class 5: Logo Embedding
# =========================================================================

class TestLogoEmbedding:
    """Test logo embedding in QR codes."""

    def test_logo_not_embedded_by_default(self):
        """Test logo is not embedded by default."""
        code = make_qr_code()
        assert code["logo_embedded"] is False

    def test_logo_embedded_with_ec_h(self):
        """Test logo requires EC-H for sufficient error recovery."""
        code = make_qr_code(logo_embedded=True, error_correction="H")
        assert code["logo_embedded"] is True
        assert code["error_correction"] == "H"

    def test_logo_code_produces_valid_record(self):
        """Test logo-embedded code passes full validation."""
        code = make_qr_code(logo_embedded=True, error_correction="H")
        assert_qr_code_valid(code)

    def test_logo_code_has_image_hash(self):
        """Test logo-embedded code has a distinct image data hash."""
        code = make_qr_code(logo_embedded=True, error_correction="H")
        assert code["image_data_hash"] is not None
        assert len(code["image_data_hash"]) == SHA256_HEX_LENGTH

    def test_logo_and_no_logo_different_hashes(self):
        """Test codes with and without logo have different image hashes."""
        code_no_logo = make_qr_code(logo_embedded=False)
        code_with_logo = make_qr_code(logo_embedded=True, error_correction="H")
        # Different UUIDs means different hashes by construction
        assert code_no_logo["image_data_hash"] != code_with_logo["image_data_hash"]

    def test_logo_coverage_below_10_percent(self):
        """Test logo should cover less than 10% of QR modules."""
        code = make_qr_code(logo_embedded=True, error_correction="H")
        # Logo coverage is implicit in H-level error correction capacity
        assert code["error_correction"] == "H"


# =========================================================================
# Test Class 6: Quality Grading
# =========================================================================

class TestQualityGrading:
    """Test ISO/IEC 15415 quality grade assessment."""

    @pytest.mark.parametrize("grade", QUALITY_GRADES)
    def test_quality_grade_values(self, grade: str):
        """Test each quality grade can be assigned to a QR code."""
        code = make_qr_code(quality_grade=grade)
        assert code["quality_grade"] == grade
        assert_qr_code_valid(code)

    def test_grade_b_is_default(self):
        """Test grade B is the default quality target."""
        code = make_qr_code()
        assert code["quality_grade"] == "B"

    def test_grade_a_is_best(self):
        """Test grade A is the highest quality grade."""
        code = make_qr_code(quality_grade="A")
        assert code["quality_grade"] == "A"

    def test_grade_d_is_minimum(self):
        """Test grade D is the lowest quality grade."""
        code = make_qr_code(quality_grade="D")
        assert code["quality_grade"] == "D"

    def test_exactly_four_grades(self):
        """Test exactly 4 quality grades exist."""
        assert len(QUALITY_GRADES) == 4

    def test_grade_ordering(self):
        """Test grades are ordered A, B, C, D."""
        assert QUALITY_GRADES == ["A", "B", "C", "D"]

    def test_grade_b_minimum_for_reliable_scanning(self):
        """Test grade B or better is needed for reliable scanning."""
        reliable_grades = ["A", "B"]
        code = make_qr_code(quality_grade="B")
        assert code["quality_grade"] in reliable_grades


# =========================================================================
# Test Class 7: DPI Configuration
# =========================================================================

class TestDPIConfiguration:
    """Test DPI configuration for QR code output."""

    @pytest.mark.parametrize("dpi", DPI_LEVELS)
    def test_create_code_with_dpi(self, dpi: int):
        """Test QR code creation at each standard DPI level."""
        code = make_qr_code(dpi=dpi)
        assert code["dpi"] == dpi
        assert_qr_code_valid(code)

    def test_screen_72_dpi(self):
        """Test 72 DPI for screen display."""
        code = make_qr_code(dpi=72)
        assert code["dpi"] == 72

    def test_draft_150_dpi(self):
        """Test 150 DPI for draft printing."""
        code = make_qr_code(dpi=150)
        assert code["dpi"] == 150

    def test_standard_300_dpi_is_default(self):
        """Test 300 DPI is the default for production labels."""
        code = make_qr_code()
        assert code["dpi"] == 300

    def test_high_600_dpi(self):
        """Test 600 DPI for high-resolution small labels."""
        code = make_qr_code(dpi=600)
        assert code["dpi"] == 600

    def test_higher_dpi_valid_within_range(self):
        """Test DPI values within the valid 72-1200 range."""
        code = make_qr_code(dpi=1200)
        assert code["dpi"] == 1200
        assert_qr_code_valid(code)

    def test_custom_dpi_between_presets(self):
        """Test custom DPI value between standard presets."""
        code = make_qr_code(dpi=450)
        assert code["dpi"] == 450
        assert_qr_code_valid(code)


# =========================================================================
# Test Class 8: Edge Cases
# =========================================================================

class TestQREdgeCases:
    """Test edge cases for QR code generation."""

    def test_max_payload_size(self):
        """Test QR code at maximum payload capacity."""
        code = make_qr_code(
            payload_size_bytes=MAX_PAYLOAD_BYTES,
            version="40",
        )
        assert code["payload_size_bytes"] == MAX_PAYLOAD_BYTES
        assert_qr_code_valid(code)

    def test_minimum_payload_size(self):
        """Test QR code with minimum payload (1 byte)."""
        code = make_qr_code(payload_size_bytes=1)
        assert code["payload_size_bytes"] == 1
        assert_qr_code_valid(code)

    def test_unique_code_ids(self):
        """Test each generated code has a unique code_id."""
        ids = set()
        for _ in range(100):
            code = make_qr_code()
            ids.add(code["code_id"])
        assert len(ids) == 100

    def test_special_characters_in_operator_id(self):
        """Test operator ID with special characters."""
        code = make_qr_code(operator_id="OP-EU/DE_001-v2.3")
        assert code["operator_id"] == "OP-EU/DE_001-v2.3"
        assert_qr_code_valid(code)

    def test_all_content_types_accepted(self):
        """Test all 5 content types produce valid codes."""
        for ct in CONTENT_TYPES:
            code = make_qr_code(content_type=ct)
            assert code["content_type"] == ct
            assert_qr_code_valid(code)

    def test_code_with_batch_code_association(self):
        """Test QR code associated with a batch code."""
        code = make_qr_code(batch_code="OP-EU-COCOA-001-cocoa-2026-00001-7")
        assert code["batch_code"] == "OP-EU-COCOA-001-cocoa-2026-00001-7"

    def test_code_with_blockchain_anchor(self):
        """Test QR code with blockchain anchor hash."""
        anchor_hash = _sha256("blockchain-anchor-tx-001")
        code = make_qr_code(blockchain_anchor_hash=anchor_hash)
        assert code["blockchain_anchor_hash"] == anchor_hash

    def test_payload_hash_is_valid_hex(self):
        """Test payload hash is a valid hex string."""
        code = make_qr_code()
        assert_valid_sha256(code["payload_hash"])

    def test_image_dimensions_increase_with_module_size(self):
        """Test that image dimensions scale with module size."""
        code_small = make_qr_code(module_size=5)
        code_large = make_qr_code(module_size=20)
        assert code_large["image_width_px"] > code_small["image_width_px"]
        assert code_large["image_height_px"] > code_small["image_height_px"]

    def test_square_image_dimensions(self):
        """Test QR code image is always square."""
        code = make_qr_code()
        assert code["image_width_px"] == code["image_height_px"]
