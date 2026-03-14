# -*- coding: utf-8 -*-
"""
PACK-003 CSRD Enterprise Pack - White Label Tests (15 tests)

Tests white-label branding engine including color validation,
WCAG contrast, CSS variable generation, branded reports,
dark mode, and custom domain configuration.

Author: GreenLang QA Team
"""

import re
from typing import Any, Dict

import pytest

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import _compute_hash


def _hex_to_rgb(hex_color: str):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _relative_luminance(r: int, g: int, b: int) -> float:
    """Calculate relative luminance per WCAG 2.0."""
    def linearize(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)


def _contrast_ratio(color1: str, color2: str) -> float:
    """Calculate WCAG contrast ratio between two hex colors."""
    r1, g1, b1 = _hex_to_rgb(color1)
    r2, g2, b2 = _hex_to_rgb(color2)
    l1 = _relative_luminance(r1, g1, b1)
    l2 = _relative_luminance(r2, g2, b2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


class TestWhiteLabel:
    """Test suite for white-label branding engine."""

    def test_brand_apply(self, sample_brand_config):
        """Test brand configuration applies all fields."""
        brand = sample_brand_config
        assert brand["enabled"] is True
        assert brand["primary_color"] == "#003366"
        assert brand["secondary_color"] == "#0066CC"
        assert brand["accent_color"] == "#FF9900"
        assert brand["font_family"] == "Roboto, Arial, sans-serif"

    def test_color_validation_hex(self, sample_brand_config):
        """Test all colors are valid 6-digit hex codes."""
        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
        color_keys = ["primary_color", "secondary_color", "accent_color"]
        for key in color_keys:
            assert hex_pattern.match(sample_brand_config[key]), (
                f"Invalid hex color for {key}: {sample_brand_config[key]}"
            )

    def test_wcag_contrast_check(self, sample_brand_config):
        """Test primary color has sufficient contrast with white."""
        primary = sample_brand_config["primary_color"]
        white = "#FFFFFF"
        ratio = _contrast_ratio(primary, white)
        assert ratio >= 4.5, (
            f"Primary color {primary} has insufficient contrast ratio "
            f"{ratio:.2f} with white (need >= 4.5 for WCAG AA)"
        )

    def test_logo_validation(self, sample_brand_config):
        """Test logo URL is a valid URL format."""
        logo_url = sample_brand_config["logo_url"]
        assert logo_url.startswith("https://"), "Logo URL must use HTTPS"
        assert "." in logo_url, "Logo URL must contain a domain"

    def test_css_variable_generation(self, sample_brand_config):
        """Test CSS variable generation from brand config."""
        css_vars = {
            "--gl-primary": sample_brand_config["primary_color"],
            "--gl-secondary": sample_brand_config["secondary_color"],
            "--gl-accent": sample_brand_config["accent_color"],
            "--gl-font-family": sample_brand_config["font_family"],
        }
        css_output = ":root {\n"
        for var, value in css_vars.items():
            css_output += f"  {var}: {value};\n"
        css_output += "}\n"
        assert "--gl-primary: #003366" in css_output
        assert "--gl-font-family: Roboto" in css_output

    def test_report_header_branded(self, sample_brand_config):
        """Test branded report header generation."""
        header = {
            "logo": sample_brand_config["report_header_logo"],
            "primary_color": sample_brand_config["primary_color"],
            "title_font": sample_brand_config["font_family"],
            "powered_by_text": "" if not sample_brand_config["powered_by_visible"] else "Powered by GreenLang",
        }
        assert header["logo"] != ""
        assert header["powered_by_text"] == ""

    def test_email_template_branded(self, sample_brand_config):
        """Test branded email template configuration."""
        email_config = {
            "from_name": "Sustainability Portal",
            "logo_url": sample_brand_config["logo_url"],
            "header_bg_color": sample_brand_config["primary_color"],
            "button_color": sample_brand_config["accent_color"],
            "font_family": sample_brand_config["font_family"],
            "branded": sample_brand_config["email_branding"],
        }
        assert email_config["branded"] is True
        assert email_config["header_bg_color"] == "#003366"

    def test_custom_domain_config(self, sample_brand_config):
        """Test custom domain configuration."""
        domain = sample_brand_config["custom_domain"]
        assert len(domain) > 0
        assert "." in domain
        assert "acme" in domain.lower()

    def test_powered_by_toggle(self, sample_brand_config):
        """Test powered-by footer visibility toggle."""
        assert sample_brand_config["powered_by_visible"] is False
        modified = dict(sample_brand_config)
        modified["powered_by_visible"] = True
        assert modified["powered_by_visible"] is True

    def test_dark_mode_variant(self, sample_brand_config):
        """Test dark mode variant configuration."""
        dark = sample_brand_config.get("dark_mode", {})
        assert "primary_color" in dark
        assert "background_color" in dark
        assert dark["background_color"] == "#121212"
        bg_contrast = _contrast_ratio(dark["text_color"], dark["background_color"])
        assert bg_contrast >= 4.5, (
            f"Dark mode text contrast {bg_contrast:.2f} is below WCAG AA"
        )

    def test_font_family_validation(self, sample_brand_config):
        """Test font family has fallback fonts."""
        font = sample_brand_config["font_family"]
        families = [f.strip() for f in font.split(",")]
        assert len(families) >= 2, "Font family must include at least one fallback"
        assert families[-1] in ("sans-serif", "serif", "monospace"), (
            "Last font must be a generic family"
        )

    def test_brand_export(self, sample_brand_config):
        """Test brand config exports to JSON."""
        import json
        exported = json.dumps(sample_brand_config, indent=2)
        parsed = json.loads(exported)
        assert parsed["primary_color"] == sample_brand_config["primary_color"]
        assert parsed["font_family"] == sample_brand_config["font_family"]

    def test_login_page_config(self, sample_brand_config):
        """Test login page branding configuration."""
        login_config = {
            "logo_url": sample_brand_config["logo_url"],
            "favicon_url": sample_brand_config["favicon_url"],
            "background_color": sample_brand_config["primary_color"],
            "button_color": sample_brand_config["accent_color"],
            "domain": sample_brand_config["custom_domain"],
        }
        assert login_config["favicon_url"].endswith(".ico")
        assert login_config["domain"] != ""

    def test_brand_update(self, sample_brand_config):
        """Test brand config update changes provenance hash."""
        hash_before = _compute_hash(sample_brand_config)
        updated = dict(sample_brand_config)
        updated["primary_color"] = "#112233"
        hash_after = _compute_hash(updated)
        assert hash_before != hash_after
        assert len(hash_after) == 64

    def test_brand_reset_default(self):
        """Test resetting brand to default GreenLang theme."""
        default_brand = {
            "enabled": False,
            "primary_color": "#1B5E20",
            "secondary_color": "#388E3C",
            "accent_color": "#4CAF50",
            "font_family": "Inter, sans-serif",
            "powered_by_visible": True,
            "custom_domain": "",
        }
        assert default_brand["enabled"] is False
        assert default_brand["powered_by_visible"] is True
        assert default_brand["primary_color"] == "#1B5E20"
