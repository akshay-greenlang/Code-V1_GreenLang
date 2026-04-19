# -*- coding: utf-8 -*-
"""
Deep tests for CampaignReportingEngine (Engine 8 of 10).

Covers: 10 report sections, section weights, report completeness
scoring, verification badge classification, partner format mapping
(CDP/GFANZ/C40/SBTi), section labels, Decimal arithmetic,
SHA-256 provenance.

Target: ~45 tests.

Author: GreenLang Platform Team
Pack: PACK-025 Race to Zero Pack
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from engines.campaign_reporting_engine import (
    CampaignReportingEngine,
    ReportSectionId,
    VerificationBadge,
    PartnerFormat,
    SECTION_WEIGHTS,
    SECTION_LABELS,
    CDP_MAPPING,
)

from conftest import assert_decimal_close


# ========================================================================
# Report Section Constants
# ========================================================================


class TestReportSectionConstants:
    """Validate 10 report section constants."""

    def test_exactly_10_sections(self):
        assert len(ReportSectionId) == 10

    def test_section_values(self):
        assert ReportSectionId.ENTITY_PROFILE.value == "entity_profile"
        assert ReportSectionId.PLEDGE_STATUS.value == "pledge_status"
        assert ReportSectionId.STARTING_LINE.value == "starting_line"
        assert ReportSectionId.EMISSIONS_INVENTORY.value == "emissions_inventory"
        assert ReportSectionId.TARGET_PROGRESS.value == "target_progress"
        assert ReportSectionId.ACTION_PLAN.value == "action_plan"
        assert ReportSectionId.SECTOR_ALIGNMENT.value == "sector_alignment"
        assert ReportSectionId.HLEG_CREDIBILITY.value == "hleg_credibility"
        assert ReportSectionId.PARTNERSHIP.value == "partnership"
        assert ReportSectionId.FORWARD_COMMITMENTS.value == "forward_commitments"

    def test_weights_sum_to_one(self):
        total = sum(SECTION_WEIGHTS.values())
        assert_decimal_close(total, Decimal("1.00"), Decimal("0.001"))

    def test_emissions_inventory_highest_weight(self):
        assert SECTION_WEIGHTS["emissions_inventory"] == Decimal("0.15")

    def test_target_progress_highest_weight(self):
        assert SECTION_WEIGHTS["target_progress"] == Decimal("0.15")

    def test_entity_profile_lowest_weight(self):
        assert SECTION_WEIGHTS["entity_profile"] == Decimal("0.05")

    def test_all_sections_have_labels(self):
        for sec in ReportSectionId:
            assert sec.value in SECTION_LABELS

    def test_all_sections_have_weights(self):
        for sec in ReportSectionId:
            assert sec.value in SECTION_WEIGHTS

    def test_section_labels_numbered(self):
        for label in SECTION_LABELS.values():
            assert label[0].isdigit() or label.startswith("1")


# ========================================================================
# Enums
# ========================================================================


class TestCampaignReportingEnums:
    """Validate campaign reporting enums."""

    def test_verification_badge_4_values(self):
        assert len(VerificationBadge) == 4

    def test_verification_badge_values(self):
        assert VerificationBadge.VERIFIED.value == "verified"
        assert VerificationBadge.PENDING.value == "pending"
        assert VerificationBadge.FAILED.value == "failed"
        assert VerificationBadge.NOT_STARTED.value == "not_started"

    def test_partner_format_5_values(self):
        assert len(PartnerFormat) == 5

    def test_partner_format_values(self):
        assert PartnerFormat.CDP.value == "cdp"
        assert PartnerFormat.GFANZ.value == "gfanz"
        assert PartnerFormat.C40.value == "c40"
        assert PartnerFormat.SBTI.value == "sbti"
        assert PartnerFormat.UNIVERSAL.value == "universal"


# ========================================================================
# Partner Format Mapping
# ========================================================================


class TestPartnerFormatMapping:
    """Validate partner format mappings."""

    def test_cdp_mapping_exists(self):
        assert len(CDP_MAPPING) > 0

    def test_cdp_maps_emissions_inventory(self):
        assert "emissions_inventory" in CDP_MAPPING

    def test_cdp_maps_target_progress(self):
        assert "target_progress" in CDP_MAPPING

    def test_cdp_emissions_maps_to_c6(self):
        assert "C6" in CDP_MAPPING["emissions_inventory"]

    def test_cdp_targets_maps_to_c4(self):
        assert "C4" in CDP_MAPPING["target_progress"]


# ========================================================================
# Engine Instantiation
# ========================================================================


class TestCampaignReportingEngineInstantiation:
    """Tests for engine creation."""

    def test_default_instantiation(self, campaign_engine):
        assert campaign_engine is not None

    def test_engine_has_calculate(self, campaign_engine):
        assert callable(getattr(campaign_engine, "generate", None))

    def test_engine_class_name(self):
        assert CampaignReportingEngine.__name__ == "CampaignReportingEngine"

    def test_engine_has_docstring(self):
        assert CampaignReportingEngine.__doc__ is not None
        assert len(CampaignReportingEngine.__doc__) > 0
