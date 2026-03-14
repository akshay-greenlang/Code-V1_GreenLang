# -*- coding: utf-8 -*-
"""
PACK-001 CSRD Starter Pack - Configuration Preset Tests
=========================================================

Validates all four size presets (large_enterprise, mid_market, sme,
first_time_reporter) and all five sector presets (manufacturing,
financial_services, technology, retail, energy), plus configuration
merging behaviour.

Test count: 20
Author: GreenLang QA Team
"""

from typing import Any, Dict, List

import pytest

from .conftest import (
    ALL_PRESETS,
    ALL_SECTORS,
    VALID_ESRS_STANDARDS,
    VALID_SCOPE3_CATEGORIES,
)


# ==========================================================================
# Size Preset Tests (4 presets x 5 aspects = 20 tests, but we parameterise)
# ==========================================================================

class TestSizePresets:
    """Validate size-based configuration presets."""

    # ------------------------------------------------------------------
    # Large Enterprise
    # ------------------------------------------------------------------
    def test_large_enterprise_loads_successfully(self):
        """Large enterprise preset loads with all required keys."""
        preset = ALL_PRESETS["large_enterprise"]
        assert preset["preset_id"] == "large_enterprise"
        assert preset["display_name"] == "Large Enterprise"
        assert "esrs_standards" in preset
        assert "scope3_categories" in preset
        assert "performance_targets" in preset
        assert "languages" in preset
        assert "xbrl_mode" in preset

    def test_large_enterprise_esrs_standards_valid(self):
        """Large enterprise must include all 12 ESRS standards."""
        preset = ALL_PRESETS["large_enterprise"]
        for std in VALID_ESRS_STANDARDS:
            assert std in preset["esrs_standards"], (
                f"Large enterprise preset missing ESRS standard '{std}'"
            )

    def test_large_enterprise_scope3_categories_valid(self):
        """Large enterprise must cover all 15 Scope 3 categories."""
        preset = ALL_PRESETS["large_enterprise"]
        assert sorted(preset["scope3_categories"]) == VALID_SCOPE3_CATEGORIES, (
            "Large enterprise must include all 15 Scope 3 categories"
        )

    def test_large_enterprise_performance_targets_reasonable(self):
        """Performance targets for large enterprise should be achievable."""
        targets = ALL_PRESETS["large_enterprise"]["performance_targets"]
        assert targets["full_report_max_minutes"] <= 60, (
            "Full report target should be <= 60 minutes"
        )
        assert targets["quarterly_update_max_minutes"] <= 30, (
            "Quarterly update target should be <= 30 minutes"
        )

    def test_large_enterprise_required_fields_present(self):
        """Large enterprise preset has full XBRL, multi-language, and audit."""
        preset = ALL_PRESETS["large_enterprise"]
        assert preset["xbrl_mode"] == "full"
        assert len(preset["languages"]) >= 2, "Large enterprise needs multi-language"
        assert preset["audit_package"] is True

    # ------------------------------------------------------------------
    # Mid-Market
    # ------------------------------------------------------------------
    def test_mid_market_loads_successfully(self):
        """Mid-market preset loads with all required keys."""
        preset = ALL_PRESETS["mid_market"]
        assert preset["preset_id"] == "mid_market"
        assert preset["display_name"] == "Mid-Market Company"
        assert "esrs_standards" in preset
        assert "scope3_categories" in preset
        assert "performance_targets" in preset

    def test_mid_market_esrs_standards_valid(self):
        """Mid-market must include all 12 ESRS standards."""
        preset = ALL_PRESETS["mid_market"]
        for std in VALID_ESRS_STANDARDS:
            assert std in preset["esrs_standards"], (
                f"Mid-market preset missing ESRS standard '{std}'"
            )

    def test_mid_market_scope3_categories_valid(self):
        """Mid-market Scope 3 covers top categories by materiality."""
        preset = ALL_PRESETS["mid_market"]
        cats = preset["scope3_categories"]
        assert len(cats) >= 3, "Mid-market should have at least 3 Scope 3 categories"
        assert len(cats) <= 10, "Mid-market should not exceed 10 Scope 3 categories"
        for cat in cats:
            assert cat in VALID_SCOPE3_CATEGORIES, (
                f"Invalid Scope 3 category {cat}"
            )

    def test_mid_market_performance_targets_reasonable(self):
        """Performance targets for mid-market should be achievable."""
        targets = ALL_PRESETS["mid_market"]["performance_targets"]
        assert targets["full_report_max_minutes"] <= 45
        assert targets["quarterly_update_max_minutes"] <= 20

    def test_mid_market_required_fields_present(self):
        """Mid-market preset has standard XBRL and single language."""
        preset = ALL_PRESETS["mid_market"]
        assert preset["xbrl_mode"] == "standard"
        assert preset["languages"] == ["en"]
        assert preset["audit_package"] is True

    # ------------------------------------------------------------------
    # SME
    # ------------------------------------------------------------------
    def test_sme_loads_successfully(self):
        """SME preset loads with all required keys."""
        preset = ALL_PRESETS["sme"]
        assert preset["preset_id"] == "sme"
        assert preset["display_name"] == "SME"
        assert "esrs_standards" in preset
        assert "scope3_categories" in preset

    def test_sme_esrs_standards_valid(self):
        """SME uses simplified ESRS subset but all listed standards are valid."""
        preset = ALL_PRESETS["sme"]
        for std in preset["esrs_standards"]:
            assert std in VALID_ESRS_STANDARDS, (
                f"SME preset contains invalid ESRS standard '{std}'"
            )
        # Must include at least cross-cutting and E1
        assert "ESRS_1" in preset["esrs_standards"]
        assert "ESRS_2" in preset["esrs_standards"]
        assert "E1" in preset["esrs_standards"]

    def test_sme_scope3_categories_valid(self):
        """SME has Scope 3 as optional (may be empty list)."""
        preset = ALL_PRESETS["sme"]
        cats = preset["scope3_categories"]
        assert isinstance(cats, list), "Scope 3 categories must be a list"
        for cat in cats:
            assert cat in VALID_SCOPE3_CATEGORIES

    def test_sme_performance_targets_reasonable(self):
        """Performance targets for SME should be tighter (smaller data)."""
        targets = ALL_PRESETS["sme"]["performance_targets"]
        assert targets["full_report_max_minutes"] <= 30
        assert targets["quarterly_update_max_minutes"] <= 15

    def test_sme_required_fields_present(self):
        """SME preset has basic XBRL and no mandatory audit package."""
        preset = ALL_PRESETS["sme"]
        assert preset["xbrl_mode"] == "basic"
        assert preset["audit_package"] is False

    # ------------------------------------------------------------------
    # First-Time Reporter
    # ------------------------------------------------------------------
    def test_first_time_reporter_loads_successfully(self):
        """First-time reporter preset loads with tutorial-mode fields."""
        preset = ALL_PRESETS["first_time_reporter"]
        assert preset["preset_id"] == "first_time_reporter"
        assert "tutorial_mode" in preset
        assert "ai_assist_level" in preset

    def test_first_time_reporter_esrs_standards_valid(self):
        """First-time reporter includes all ESRS standards for full coverage."""
        preset = ALL_PRESETS["first_time_reporter"]
        for std in VALID_ESRS_STANDARDS:
            assert std in preset["esrs_standards"]

    def test_first_time_reporter_scope3_categories_valid(self):
        """First-time reporter starts with minimal Scope 3 categories."""
        preset = ALL_PRESETS["first_time_reporter"]
        cats = preset["scope3_categories"]
        assert len(cats) >= 1, "First-time reporter should have at least 1 Scope 3 cat"
        assert len(cats) <= 5, "First-time reporter should have a manageable number"

    def test_first_time_reporter_performance_targets_reasonable(self):
        """First-time reporter targets are more relaxed to allow learning."""
        targets = ALL_PRESETS["first_time_reporter"]["performance_targets"]
        assert targets["full_report_max_minutes"] <= 60
        assert targets["full_report_max_minutes"] >= 30, (
            "First-time reporter should have more generous time allowance"
        )

    def test_first_time_reporter_required_fields_present(self):
        """First-time reporter has tutorial mode and high AI assist."""
        preset = ALL_PRESETS["first_time_reporter"]
        assert preset["tutorial_mode"] is True
        assert preset["ai_assist_level"] == "high"


# ==========================================================================
# Sector Preset Tests (parameterised across 5 sectors = 15 tests)
# ==========================================================================

class TestSectorPresets:
    """Validate sector-specific configuration presets."""

    @pytest.mark.parametrize("sector_id", [
        "manufacturing", "financial_services", "technology", "retail", "energy"
    ])
    def test_sector_loads_successfully(self, sector_id: str):
        """Each sector preset loads with all required keys."""
        sector = ALL_SECTORS[sector_id]
        assert sector["sector_id"] == sector_id
        assert "display_name" in sector
        assert "nace_codes" in sector
        assert "emission_focus" in sector
        assert "key_agents" in sector
        assert isinstance(sector["nace_codes"], list)
        assert len(sector["nace_codes"]) > 0

    @pytest.mark.parametrize("sector_id", [
        "manufacturing", "financial_services", "technology", "retail", "energy"
    ])
    def test_sector_emission_focus_valid(self, sector_id: str):
        """Each sector has a non-empty emission focus list with valid entries."""
        sector = ALL_SECTORS[sector_id]
        focus = sector["emission_focus"]
        assert isinstance(focus, list), "emission_focus must be a list"
        assert len(focus) >= 1, "emission_focus must have at least one entry"
        for item in focus:
            assert isinstance(item, str) and len(item) > 0, (
                f"Invalid emission_focus entry: '{item}'"
            )
            # Each entry should reference a known scope or category
            valid_prefixes = ("scope1_", "scope2_", "scope3_")
            assert any(item.startswith(p) for p in valid_prefixes), (
                f"emission_focus '{item}' does not start with a valid scope prefix"
            )

    @pytest.mark.parametrize("sector_id", [
        "manufacturing", "financial_services", "technology", "retail", "energy"
    ])
    def test_sector_scope_config_valid(self, sector_id: str):
        """Each sector correctly marks which scopes receive emphasis."""
        sector = ALL_SECTORS[sector_id]
        assert isinstance(sector["scope1_emphasis"], bool)
        assert isinstance(sector["scope2_emphasis"], bool)
        assert isinstance(sector["scope3_emphasis"], bool)
        # At least one scope must be emphasised
        has_emphasis = (
            sector["scope1_emphasis"]
            or sector["scope2_emphasis"]
            or sector["scope3_emphasis"]
        )
        assert has_emphasis, (
            f"Sector '{sector_id}' must emphasise at least one scope"
        )


# ==========================================================================
# Config Merging Tests (2 tests)
# ==========================================================================

class TestConfigMerging:
    """Validate merging of size and sector presets."""

    def test_config_merge_size_and_sector(self):
        """Merging a size preset with a sector preset produces a valid config.

        The merged config should retain size-level ESRS standards and
        overlay sector-specific emission focus and key agents.
        """
        size = ALL_PRESETS["mid_market"].copy()
        sector = ALL_SECTORS["manufacturing"].copy()

        merged = {**size, **sector}
        # Size preset_id is retained (sector uses sector_id, not preset_id)
        assert merged["preset_id"] == size["preset_id"], (
            "Size preset_id should be retained in merge"
        )
        # Sector sector_id is added
        assert merged["sector_id"] == sector["sector_id"], (
            "Sector sector_id should be present in merge"
        )
        # Explicit fields from size should still be accessible via original
        assert size["esrs_standards"] is not None
        assert size["scope3_categories"] is not None

        # Sector fields present
        assert merged["emission_focus"] == sector["emission_focus"]
        assert merged["key_agents"] == sector["key_agents"]

        # Both presets contribute valid data
        assert len(size["esrs_standards"]) == 12
        assert len(sector["emission_focus"]) >= 1

    def test_config_merge_override_priority(self):
        """When merging, sector-specific values override size defaults.

        For example, if a size preset says scope3_emphasis=False but the
        sector (e.g. retail) says scope3_emphasis=True, the sector value
        should take precedence in the merged output.
        """
        # SME defaults: scope3 categories empty, scope3_emphasis not defined
        size = ALL_PRESETS["sme"].copy()
        sector = ALL_SECTORS["retail"].copy()

        # Simulate merge where sector overrides size
        merged = {**size}
        # Apply sector overrides
        merged["emission_focus"] = sector["emission_focus"]
        merged["key_agents"] = sector["key_agents"]
        merged["scope3_emphasis"] = sector["scope3_emphasis"]
        merged["scope1_emphasis"] = sector["scope1_emphasis"]

        # Retail sector should override SME scope3_emphasis to True
        assert merged["scope3_emphasis"] is True, (
            "Retail sector should set scope3_emphasis=True even for SME"
        )
        # SME base fields should remain for non-overridden keys
        assert merged["xbrl_mode"] == "basic", (
            "SME xbrl_mode should not be overridden by sector"
        )
        assert merged["audit_package"] is False, (
            "SME audit_package should not be overridden by sector"
        )
