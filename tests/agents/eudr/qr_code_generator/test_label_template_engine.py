# -*- coding: utf-8 -*-
"""
Unit tests for Engine 3: Label Template Engine (AGENT-EUDR-014)

Tests label rendering including label templates, compliance colours,
label output formats, multi-QR labels, label dimensions, font
configuration, print bleed, and edge cases.

55+ tests across 8 test classes.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014 QR Code Generator (GL-EUDR-QRG-014)
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from .conftest import (
    COMPLIANCE_STATUSES,
    COMPLIANT_COLOR_HEX,
    DPI_LEVELS,
    LABEL_DIMENSIONS,
    LABEL_TEMPLATES,
    NON_COMPLIANT_COLOR_HEX,
    PENDING_COLOR_HEX,
    SAMPLE_CODE_ID,
    SAMPLE_CODE_ID_2,
    SAMPLE_COMMODITY,
    SAMPLE_OPERATOR_ID,
    SAMPLE_PRODUCT_NAME,
    SAMPLE_VERIFICATION_URL,
    SHA256_HEX_LENGTH,
    SUPPORTED_FORMATS,
    assert_label_valid,
    make_label,
)


# =========================================================================
# Test Class 1: Label Templates
# =========================================================================

class TestLabelTemplates:
    """Test label rendering for all 5 template types."""

    @pytest.mark.parametrize("template", LABEL_TEMPLATES)
    def test_create_label_with_template(self, template: str):
        """Test label creation for each template type."""
        label = make_label(template=template)
        assert label["template"] == template
        assert_label_valid(label)

    def test_product_label_is_default(self):
        """Test product label is the default template."""
        label = make_label()
        assert label["template"] == "product_label"

    def test_shipping_label_includes_batch_code(self):
        """Test shipping label can include batch code."""
        label = make_label(
            template="shipping_label",
            batch_code="OP-EU-COCOA-001-cocoa-2026-00001-7",
        )
        assert label["batch_code"] is not None
        assert len(label["batch_code"]) > 0

    def test_pallet_label_large_dimensions(self):
        """Test pallet label has larger dimensions."""
        label = make_label(template="pallet_label")
        assert label["width_mm"] == 150.0
        assert label["height_mm"] == 200.0

    def test_container_label_largest(self):
        """Test container label has the largest dimensions."""
        label = make_label(template="container_label")
        assert label["width_mm"] == 200.0
        assert label["height_mm"] == 250.0

    def test_consumer_label_smallest(self):
        """Test consumer label has the smallest dimensions."""
        label = make_label(template="consumer_label")
        assert label["width_mm"] == 40.0
        assert label["height_mm"] == 25.0

    def test_exactly_five_templates(self):
        """Test exactly 5 label templates are supported."""
        assert len(LABEL_TEMPLATES) == 5


# =========================================================================
# Test Class 2: Compliance Colours
# =========================================================================

class TestComplianceColors:
    """Test EUDR compliance status colour mapping."""

    def test_compliant_green_color(self):
        """Test compliant status maps to green colour."""
        label = make_label(compliance_status="compliant")
        assert label["compliance_color_hex"] == COMPLIANT_COLOR_HEX

    def test_pending_amber_color(self):
        """Test pending status maps to amber colour."""
        label = make_label(compliance_status="pending")
        assert label["compliance_color_hex"] == PENDING_COLOR_HEX

    def test_non_compliant_red_color(self):
        """Test non-compliant status maps to red colour."""
        label = make_label(compliance_status="non_compliant")
        assert label["compliance_color_hex"] == NON_COMPLIANT_COLOR_HEX

    def test_under_review_amber_color(self):
        """Test under review status maps to amber colour."""
        label = make_label(compliance_status="under_review")
        assert label["compliance_color_hex"] == PENDING_COLOR_HEX

    def test_color_hex_format(self):
        """Test all colours are in #RRGGBB format."""
        for status in COMPLIANCE_STATUSES:
            label = make_label(compliance_status=status)
            color = label["compliance_color_hex"]
            assert color.startswith("#"), f"Color must start with # for {status}"
            assert len(color) == 7, f"Color must be #RRGGBB for {status}"

    @pytest.mark.parametrize("status", COMPLIANCE_STATUSES)
    def test_all_compliance_statuses_have_colors(self, status: str):
        """Test every compliance status gets a colour assigned."""
        label = make_label(compliance_status=status)
        assert label["compliance_color_hex"] is not None
        assert len(label["compliance_color_hex"]) == 7

    def test_compliant_color_constant(self):
        """Test compliant colour constant matches expected value."""
        assert COMPLIANT_COLOR_HEX == "#2E7D32"

    def test_non_compliant_color_constant(self):
        """Test non-compliant colour constant matches expected value."""
        assert NON_COMPLIANT_COLOR_HEX == "#C62828"


# =========================================================================
# Test Class 3: Label Output Formats
# =========================================================================

class TestLabelOutputFormats:
    """Test label rendering in different output formats."""

    @pytest.mark.parametrize("fmt", ["png", "svg", "pdf", "zpl"])
    def test_label_output_format(self, fmt: str):
        """Test label rendering in common output formats."""
        label = make_label(output_format=fmt)
        assert label["output_format"] == fmt
        assert_label_valid(label)

    def test_pdf_is_default_label_format(self):
        """Test PDF is the default label output format."""
        label = make_label()
        assert label["output_format"] == "pdf"

    def test_zpl_for_thermal_printers(self):
        """Test ZPL output for thermal printer labels."""
        label = make_label(output_format="zpl")
        assert label["output_format"] == "zpl"

    def test_png_for_digital_display(self):
        """Test PNG output for digital display."""
        label = make_label(output_format="png")
        assert label["output_format"] == "png"

    def test_svg_for_scalable_labels(self):
        """Test SVG output for scalable vector labels."""
        label = make_label(output_format="svg")
        assert label["output_format"] == "svg"

    def test_label_has_file_size(self):
        """Test rendered label has a file size."""
        label = make_label()
        assert label["file_size_bytes"] is not None
        assert label["file_size_bytes"] > 0

    def test_label_has_image_hash(self):
        """Test rendered label has an image data hash."""
        label = make_label()
        assert label["image_data_hash"] is not None
        assert len(label["image_data_hash"]) == SHA256_HEX_LENGTH


# =========================================================================
# Test Class 4: Multi-QR Labels
# =========================================================================

class TestMultiQRLabels:
    """Test labels containing multiple QR codes."""

    def test_label_with_single_qr(self):
        """Test standard label with one QR code."""
        label = make_label(code_id=SAMPLE_CODE_ID)
        assert label["code_id"] == SAMPLE_CODE_ID

    def test_label_with_different_code_id(self):
        """Test label can reference different code IDs."""
        label1 = make_label(code_id=SAMPLE_CODE_ID)
        label2 = make_label(code_id=SAMPLE_CODE_ID_2)
        assert label1["code_id"] != label2["code_id"]

    def test_shipping_label_with_batch_and_qr(self):
        """Test shipping label with both batch code and QR code."""
        label = make_label(
            template="shipping_label",
            batch_code="BATCH-001",
            verification_url=SAMPLE_VERIFICATION_URL,
        )
        assert label["batch_code"] is not None
        assert label["verification_url"] is not None

    def test_pallet_label_with_multiple_refs(self):
        """Test pallet label referencing multiple identifiers."""
        label = make_label(
            template="pallet_label",
            batch_code="BATCH-PALLET-001",
            custom_fields={
                "sscc": "00012345678901234567",
                "handling_instruction": "KEEP DRY",
            },
        )
        assert label["custom_fields"]["sscc"] == "00012345678901234567"

    def test_container_label_with_customs_ref(self):
        """Test container label with customs reference."""
        label = make_label(
            template="container_label",
            custom_fields={
                "container_number": "MSCU1234567",
                "seal_number": "SEAL-001",
                "customs_ref": "MRN-2026-EU-001",
            },
        )
        assert label["custom_fields"]["container_number"] == "MSCU1234567"

    def test_unique_label_ids(self):
        """Test each generated label has a unique label_id."""
        ids = {make_label()["label_id"] for _ in range(50)}
        assert len(ids) == 50


# =========================================================================
# Test Class 5: Label Dimensions
# =========================================================================

class TestLabelDimensions:
    """Test correct dimensions per template type."""

    @pytest.mark.parametrize("template", LABEL_TEMPLATES)
    def test_template_has_dimensions(self, template: str):
        """Test each template type has defined dimensions."""
        label = make_label(template=template)
        assert label["width_mm"] is not None
        assert label["width_mm"] > 0
        assert label["height_mm"] is not None
        assert label["height_mm"] > 0

    @pytest.mark.parametrize("template,expected", [
        ("product_label", {"width_mm": 50.0, "height_mm": 30.0}),
        ("shipping_label", {"width_mm": 100.0, "height_mm": 150.0}),
        ("pallet_label", {"width_mm": 150.0, "height_mm": 200.0}),
        ("container_label", {"width_mm": 200.0, "height_mm": 250.0}),
        ("consumer_label", {"width_mm": 40.0, "height_mm": 25.0}),
    ])
    def test_template_dimensions_match(self, template: str, expected: Dict[str, float]):
        """Test each template type returns correct dimensions."""
        label = make_label(template=template)
        assert label["width_mm"] == expected["width_mm"]
        assert label["height_mm"] == expected["height_mm"]

    def test_label_dimensions_constant_dict(self):
        """Test LABEL_DIMENSIONS constant has all templates."""
        assert len(LABEL_DIMENSIONS) == 5
        for template in LABEL_TEMPLATES:
            assert template in LABEL_DIMENSIONS

    def test_custom_dimensions_via_override(self):
        """Test label dimensions can be overridden."""
        label = make_label(width_mm=80.0, height_mm=60.0)
        assert label["width_mm"] == 80.0
        assert label["height_mm"] == 60.0

    def test_container_largest_width(self):
        """Test container label has the largest width."""
        widths = {t: LABEL_DIMENSIONS[t]["width_mm"] for t in LABEL_TEMPLATES}
        assert widths["container_label"] == max(widths.values())


# =========================================================================
# Test Class 6: Font Configuration
# =========================================================================

class TestFontConfiguration:
    """Test font family, size, and multi-language support."""

    def test_default_font_family(self):
        """Test default font is DejaVuSans."""
        label = make_label()
        assert label["font"] == "DejaVuSans"

    def test_custom_font_family(self):
        """Test custom font family can be set."""
        label = make_label(font="NotoSans")
        assert label["font"] == "NotoSans"

    def test_default_font_size(self):
        """Test default font size is 12pt."""
        label = make_label()
        assert label["font_size"] == 12

    def test_minimum_font_size(self):
        """Test minimum font size of 4pt."""
        label = make_label(font_size=4)
        assert label["font_size"] == 4

    def test_maximum_font_size(self):
        """Test maximum font size of 72pt."""
        label = make_label(font_size=72)
        assert label["font_size"] == 72

    def test_consumer_label_smaller_font(self):
        """Test consumer label uses smaller font size."""
        label = make_label(template="consumer_label", font_size=8)
        assert label["font_size"] == 8

    def test_multi_language_font(self):
        """Test font supporting multiple scripts."""
        label = make_label(font="NotoSansCJK")
        assert label["font"] == "NotoSansCJK"
        assert_label_valid(label)


# =========================================================================
# Test Class 7: Print Bleed
# =========================================================================

class TestPrintBleed:
    """Test print bleed margins and safe areas."""

    def test_default_bleed_3mm(self):
        """Test default print bleed is 3mm."""
        label = make_label()
        assert label["bleed_mm"] == 3

    def test_no_bleed(self):
        """Test label with zero bleed for digital display."""
        label = make_label(bleed_mm=0)
        assert label["bleed_mm"] == 0

    def test_large_bleed_for_offset_printing(self):
        """Test larger bleed for offset printing."""
        label = make_label(bleed_mm=5)
        assert label["bleed_mm"] == 5

    def test_bleed_does_not_exceed_label_width(self):
        """Test bleed is smaller than label dimensions."""
        label = make_label(template="consumer_label")
        assert label["bleed_mm"] < label["width_mm"]
        assert label["bleed_mm"] < label["height_mm"]

    def test_bleed_is_non_negative(self):
        """Test bleed cannot be negative."""
        label = make_label()
        assert label["bleed_mm"] >= 0

    @pytest.mark.parametrize("bleed", [0, 1, 2, 3, 5, 10])
    def test_various_bleed_values(self, bleed: int):
        """Test various bleed margin values."""
        label = make_label(bleed_mm=bleed)
        assert label["bleed_mm"] == bleed
        assert_label_valid(label)


# =========================================================================
# Test Class 8: Edge Cases
# =========================================================================

class TestLabelEdgeCases:
    """Test edge cases for label rendering."""

    def test_custom_template_fields(self):
        """Test label with custom fields added to template."""
        label = make_label(
            custom_fields={"certification": "Rainforest Alliance", "lot": "LOT-2026"},
        )
        assert label["custom_fields"]["certification"] == "Rainforest Alliance"
        assert label["custom_fields"]["lot"] == "LOT-2026"

    def test_empty_custom_fields(self):
        """Test label with empty custom fields dict."""
        label = make_label()
        assert label["custom_fields"] == {}

    def test_large_label_dimensions(self):
        """Test label with very large dimensions."""
        label = make_label(width_mm=500.0, height_mm=500.0)
        assert label["width_mm"] == 500.0
        assert label["height_mm"] == 500.0

    def test_minimal_label_dimensions(self):
        """Test label with very small dimensions."""
        label = make_label(width_mm=10.0, height_mm=10.0)
        assert label["width_mm"] == 10.0
        assert label["height_mm"] == 10.0

    def test_label_with_verification_url(self):
        """Test label includes verification URL."""
        label = make_label(verification_url=SAMPLE_VERIFICATION_URL)
        assert label["verification_url"] == SAMPLE_VERIFICATION_URL

    def test_label_without_product_name(self):
        """Test label without product name (batch labels)."""
        label = make_label(product_name=None)
        assert label["product_name"] is None

    def test_label_with_long_product_name(self):
        """Test label with a long product name."""
        long_name = "A" * 200
        label = make_label(product_name=long_name)
        assert label["product_name"] == long_name

    @pytest.mark.parametrize("dpi", DPI_LEVELS)
    def test_label_dpi_levels(self, dpi: int):
        """Test label at each standard DPI level."""
        label = make_label(dpi=dpi)
        assert label["dpi"] == dpi
        assert_label_valid(label)
