# -*- coding: utf-8 -*-
"""Tests for the EU Battery Regulation method pack.

Covers:
* class-threshold enforcement per Regulation (EU) 2023/1542 Article 3(9)-(14)
* CFP declaration obligations per Article 7(1) indents
* classification helper for all five battery classes
* DPP-compatible audit output format with all required substitutions
"""
from __future__ import annotations

import pytest

from greenlang.data.canonical_v2 import (
    FactorFamily,
    FormulaType,
    MethodProfile,
)
from greenlang.factors.method_packs.eu_policy import (
    EU_BATTERY,
    EU_BATTERY_CLASS_THRESHOLDS,
    BatteryClass,
    BatteryClassThreshold,
    cfp_declaration_required,
    classify_battery,
    get_battery_threshold,
)
from greenlang.factors.method_packs.registry import get_pack


# ---------------------------------------------------------------------------
# Pack registration and profile
# ---------------------------------------------------------------------------


class TestPackRegistration:
    def test_pack_registered_under_eu_dpp_battery_profile(self):
        pack = get_pack(MethodProfile.EU_DPP_BATTERY)
        assert pack is EU_BATTERY
        assert pack.profile is MethodProfile.EU_DPP_BATTERY

    def test_pack_requires_certified_and_verified(self):
        rule = EU_BATTERY.selection_rule
        assert rule.allowed_statuses == ("certified",)
        assert rule.require_verification is True

    def test_pack_reporting_labels_include_article_7(self):
        assert "Article_7_CFP" in EU_BATTERY.reporting_labels
        assert "EU_Battery_Regulation" in EU_BATTERY.reporting_labels
        assert "EU_DPP" in EU_BATTERY.reporting_labels

    def test_pack_allowed_families_cover_battery_lifecycle(self):
        allowed = {f for f in EU_BATTERY.selection_rule.allowed_families}
        assert FactorFamily.MATERIAL_EMBODIED in allowed
        assert FactorFamily.EMISSIONS in allowed
        assert FactorFamily.GRID_INTENSITY in allowed

    def test_pack_allowed_formula_types_include_lca(self):
        allowed = {t for t in EU_BATTERY.selection_rule.allowed_formula_types}
        assert FormulaType.LCA in allowed


# ---------------------------------------------------------------------------
# Class threshold table
# ---------------------------------------------------------------------------


class TestBatteryClassThresholds:
    def test_all_five_classes_present(self):
        classes = {row.battery_class for row in EU_BATTERY_CLASS_THRESHOLDS}
        assert classes == {
            BatteryClass.PORTABLE,
            BatteryClass.LMT,
            BatteryClass.INDUSTRIAL_STATIONARY,
            BatteryClass.EV,
            BatteryClass.ELECTRIC_AVIATION,
        }

    def test_portable_max_weight_5kg(self):
        row = get_battery_threshold(BatteryClass.PORTABLE)
        assert row.max_weight_kg == 5.0
        # Not mandated in Art 7 v1
        assert row.cfp_declaration_required is False

    def test_lmt_energy_window_0_025_to_2_kwh(self):
        row = get_battery_threshold(BatteryClass.LMT)
        assert row.min_energy_kwh == 0.025
        assert row.max_energy_kwh == 2.0
        assert row.cfp_declaration_required is True
        assert row.enforcement_date_iso == "2028-08-18"

    def test_industrial_stationary_above_2_kwh(self):
        row = get_battery_threshold(BatteryClass.INDUSTRIAL_STATIONARY)
        assert row.min_energy_kwh == 2.0
        assert row.max_energy_kwh is None
        assert row.cfp_declaration_required is True
        assert row.enforcement_date_iso == "2024-02-18"

    def test_ev_declaration_required_from_2024(self):
        row = get_battery_threshold(BatteryClass.EV)
        assert row.cfp_declaration_required is True
        assert row.enforcement_date_iso == "2024-02-18"
        assert row.dpp_required is True

    def test_electric_aviation_declaration_deferred(self):
        row = get_battery_threshold(BatteryClass.ELECTRIC_AVIATION)
        # Not yet mandated — pending Commission implementing act
        assert row.cfp_declaration_required is False

    def test_threshold_has_regulation_citation(self):
        for row in EU_BATTERY_CLASS_THRESHOLDS:
            assert "2023/1542" in row.legal_article

    def test_thresholds_immutable(self):
        row = get_battery_threshold(BatteryClass.LMT)
        with pytest.raises(Exception):
            # frozen dataclass -> mutation blocked
            row.max_energy_kwh = 5.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# classify_battery()
# ---------------------------------------------------------------------------


class TestClassifyBattery:
    @pytest.mark.parametrize(
        "weight,energy,expected",
        [
            # Portable: <= 5kg, no qualifying energy
            (0.2, None, BatteryClass.PORTABLE),                    # AA cell
            (4.9, 0.02, BatteryClass.PORTABLE),                    # just below LMT floor
            # LMT: 0.025 - 2 kWh inclusive
            (3.0, 0.025, BatteryClass.LMT),                        # e-scooter floor
            (20.0, 1.0, BatteryClass.LMT),                         # e-bike battery
            (35.0, 2.0, BatteryClass.LMT),                         # e-cargo cycle ceiling
            # Industrial stationary: > 2 kWh, no mobility
            (120.0, 10.0, BatteryClass.INDUSTRIAL_STATIONARY),     # home ESS
            (2500.0, 500.0, BatteryClass.INDUSTRIAL_STATIONARY),   # grid BESS
        ],
    )
    def test_classification_cases(self, weight, energy, expected):
        assert classify_battery(weight_kg=weight, energy_kwh=energy) is expected

    def test_classify_requires_at_least_one_dimension(self):
        with pytest.raises(ValueError, match="at least one"):
            classify_battery(weight_kg=None, energy_kwh=None)

    def test_lmt_window_inclusive_at_both_ends(self):
        # Exactly 0.025 kWh -> LMT (Art 3(11) ">= 0.025")
        assert classify_battery(weight_kg=5.0, energy_kwh=0.025) is BatteryClass.LMT
        # Exactly 2.0 kWh -> LMT (Art 3(11) "<= 2")
        assert classify_battery(weight_kg=30.0, energy_kwh=2.0) is BatteryClass.LMT

    def test_industrial_stationary_strictly_above_2kwh(self):
        # 2.01 kWh is just above -> industrial
        assert (
            classify_battery(weight_kg=50.0, energy_kwh=2.01)
            is BatteryClass.INDUSTRIAL_STATIONARY
        )


# ---------------------------------------------------------------------------
# cfp_declaration_required()
# ---------------------------------------------------------------------------


class TestCFPDeclaration:
    def test_ev_100_kwh_requires_declaration(self):
        # A typical EV pack (60-100 kWh) must carry a CFP declaration
        # since 2024-02-18.
        assert cfp_declaration_required(weight_kg=500.0, energy_kwh=100.0) is True

    def test_lmt_1_kwh_requires_declaration(self):
        # E-bike 1 kWh -> LMT -> CFP declaration from 2028-08-18
        assert cfp_declaration_required(weight_kg=20.0, energy_kwh=1.0) is True

    def test_portable_aa_does_not_require_declaration(self):
        # Small consumer portable battery -> no CFP declaration in v1.
        assert cfp_declaration_required(weight_kg=0.1, energy_kwh=None) is False

    def test_industrial_10kwh_requires_declaration(self):
        assert cfp_declaration_required(weight_kg=100.0, energy_kwh=10.0) is True


# ---------------------------------------------------------------------------
# DPP-compatible audit output format
# ---------------------------------------------------------------------------


class TestDPPAuditFormat:
    def test_audit_template_has_battery_class_slot(self):
        template = EU_BATTERY.audit_text_template
        # Required slots for a DPP-compatible CFP declaration
        assert "{battery_class}" in template
        assert "{battery_energy_kwh}" in template
        assert "{battery_weight_kg}" in template
        assert "{enforcement_date}" in template
        assert "{dpp_id}" in template
        assert "{factor_id}" in template
        assert "{source_org}" in template
        assert "{verification_status}" in template

    def test_audit_template_references_functional_unit(self):
        # EU 2023/1542 Annex II calls for functional unit = 1 kWh of
        # energy delivered over service life.
        template = EU_BATTERY.audit_text_template
        assert "1 kWh" in template
        assert "service life" in template

    def test_audit_renders_for_ev_class(self):
        row = get_battery_threshold(BatteryClass.EV)
        rendered = EU_BATTERY.audit_text_template.format(
            battery_class=BatteryClass.EV.value,
            battery_energy_kwh=75.0,
            battery_weight_kg=450.0,
            enforcement_date=row.enforcement_date_iso,
            dpp_id="DPP-ACME-EV-001",
            factor_id="EF:BATTERY:NMC811:2026",
            source_org="Acme Gigafactory",
            source_year=2026,
            verification_status="external_verified",
        )
        assert "ev" in rendered.lower()
        assert "2024-02-18" in rendered
        assert "DPP-ACME-EV-001" in rendered
        assert "EF:BATTERY:NMC811:2026" in rendered

    def test_pack_tagged_licensed_and_dpp(self):
        assert "licensed" in EU_BATTERY.tags
        assert "dpp" in EU_BATTERY.tags
        assert "battery" in EU_BATTERY.tags
