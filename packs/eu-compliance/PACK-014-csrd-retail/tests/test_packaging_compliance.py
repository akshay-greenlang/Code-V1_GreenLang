# -*- coding: utf-8 -*-
"""
Unit tests for PackagingComplianceEngine -- PACK-014 CSRD Retail Engine 3
==========================================================================

Tests PPWR packaging compliance including recycled content targets,
EPR eco-modulation fees, labeling requirements, reuse targets,
substance restrictions, and portfolio-level compliance scoring.

Coverage target: 85%+
Total tests: ~38
"""

import importlib.util
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Dynamic module loading
# ---------------------------------------------------------------------------
ENGINES_DIR = os.path.join(os.path.dirname(__file__), "..", "engines")


def _load(name: str):
    path = os.path.join(ENGINES_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_m = _load("packaging_compliance_engine")

PackagingComplianceEngine = _m.PackagingComplianceEngine
PackagingItem = _m.PackagingItem
PackagingPortfolio = _m.PackagingPortfolio
PackagingMaterial = _m.PackagingMaterial
PackagingType = _m.PackagingType
EPRGrade = _m.EPRGrade
LabelingStatus = _m.LabelingStatus
PPWRComplianceResult = _m.PPWRComplianceResult
RecycledContentAssessment = _m.RecycledContentAssessment
EPRFeeDetail = _m.EPRFeeDetail
LabelingComplianceDetail = _m.LabelingComplianceDetail
ReuseProgressDetail = _m.ReuseProgressDetail
SubstanceComplianceDetail = _m.SubstanceComplianceDetail
CarbonFootprintSummary = _m.CarbonFootprintSummary
PPWR_RECYCLED_CONTENT_TARGETS = _m.PPWR_RECYCLED_CONTENT_TARGETS
PPWR_REUSE_TARGETS = _m.PPWR_REUSE_TARGETS
EPR_GRADE_MULTIPLIERS = _m.EPR_GRADE_MULTIPLIERS
EPR_BASE_RATES = _m.EPR_BASE_RATES
LABELING_REQUIREMENTS = _m.LABELING_REQUIREMENTS
SUBSTANCE_RESTRICTIONS = _m.SUBSTANCE_RESTRICTIONS
PACKAGING_CARBON_FACTORS = _m.PACKAGING_CARBON_FACTORS
RECYCLED_CARBON_REDUCTION = _m.RECYCLED_CARBON_REDUCTION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(**overrides) -> PackagingItem:
    """Build a minimal valid PackagingItem with optional overrides."""
    defaults = dict(
        item_id="PKG001",
        item_name="Test Bottle",
        material=PackagingMaterial.PET,
        packaging_type=PackagingType.PRIMARY,
        weight_grams=30.0,
        units_placed=100_000,
        recycled_content_pct=0.0,
        recyclability_grade=EPRGrade.C,
    )
    defaults.update(overrides)
    return PackagingItem(**defaults)


def _make_portfolio(items=None, **overrides) -> PackagingPortfolio:
    """Build a PackagingPortfolio from items list."""
    if items is None:
        items = [_make_item()]
    defaults = dict(
        organisation_id="ORG001",
        reporting_year=2026,
        items=items,
    )
    defaults.update(overrides)
    return PackagingPortfolio(**defaults)


# ===================================================================
# TestInitialization
# ===================================================================


class TestInitialization:
    """Engine instantiation tests."""

    def test_default(self):
        engine = PackagingComplianceEngine()
        assert engine is not None

    def test_config_rc_targets(self):
        engine = PackagingComplianceEngine()
        assert len(engine._rc_targets) > 0

    def test_config_epr_rates(self):
        engine = PackagingComplianceEngine()
        assert PackagingMaterial.PET in engine._epr_base_rates

    def test_config_substance_limits(self):
        engine = PackagingComplianceEngine()
        assert "heavy_metals" in engine._substance_limits


# ===================================================================
# TestPackagingMaterials
# ===================================================================


class TestPackagingMaterials:
    """PackagingMaterial enum tests."""

    def test_all_12_defined(self):
        assert len(PackagingMaterial) == 12

    def test_enum_values(self):
        expected = {
            "PET", "HDPE", "PP", "PS", "PVC", "glass",
            "aluminium", "steel", "paper_board", "wood",
            "composite", "bioplastic",
        }
        actual = {m.value for m in PackagingMaterial}
        assert actual == expected

    def test_pet(self):
        assert PackagingMaterial.PET.value == "PET"

    def test_paper_board(self):
        assert PackagingMaterial.PAPER_BOARD.value == "paper_board"


# ===================================================================
# TestRecycledContent
# ===================================================================


class TestRecycledContent:
    """Recycled content assessment tests."""

    def test_pet_30pct_target_2030(self):
        targets = PPWR_RECYCLED_CONTENT_TARGETS["PET"]
        assert targets[2030] == 30.0

    def test_gap_calculation(self):
        engine = PackagingComplianceEngine()
        item = _make_item(
            material=PackagingMaterial.PET,
            recycled_content_pct=20.0,
            is_contact_sensitive=True,
        )
        portfolio = _make_portfolio(items=[item])
        result = engine.assess_compliance(portfolio)
        pet_rc = [r for r in result.recycled_content_by_material if r.material == "PET"][0]
        # Gap to 2030: 30 - 20 = 10
        assert pet_rc.gap_to_2030_pct == pytest.approx(10.0, abs=1.0)

    def test_weighted_average(self):
        engine = PackagingComplianceEngine()
        items = [
            _make_item(item_id="A", weight_grams=50.0, units_placed=1000, recycled_content_pct=40.0),
            _make_item(item_id="B", weight_grams=50.0, units_placed=1000, recycled_content_pct=20.0),
        ]
        portfolio = _make_portfolio(items=items)
        result = engine.assess_compliance(portfolio)
        # Equal weight -> average = 30%
        assert result.overall_recycled_content_pct == pytest.approx(30.0, abs=1.0)

    def test_above_target(self):
        engine = PackagingComplianceEngine()
        item = _make_item(
            material=PackagingMaterial.PET,
            recycled_content_pct=50.0,
            is_contact_sensitive=True,
        )
        portfolio = _make_portfolio(items=[item])
        result = engine.assess_compliance(portfolio)
        pet_rc = [r for r in result.recycled_content_by_material if r.material == "PET"][0]
        assert pet_rc.compliant_2030 is True

    def test_multiple_materials(self):
        engine = PackagingComplianceEngine()
        items = [
            _make_item(item_id="A", material=PackagingMaterial.PET, recycled_content_pct=35.0),
            _make_item(item_id="B", material=PackagingMaterial.HDPE, recycled_content_pct=5.0),
        ]
        portfolio = _make_portfolio(items=items)
        result = engine.assess_compliance(portfolio)
        assert len(result.recycled_content_by_material) == 2


# ===================================================================
# TestEPRGrades
# ===================================================================


class TestEPRGrades:
    """EPR eco-modulation fee tests."""

    def test_grade_a_multiplier(self):
        assert EPR_GRADE_MULTIPLIERS[EPRGrade.A] == 0.50

    def test_grade_e_multiplier(self):
        assert EPR_GRADE_MULTIPLIERS[EPRGrade.E] == 2.00

    def test_fee_calculation(self):
        engine = PackagingComplianceEngine()
        item = _make_item(
            material=PackagingMaterial.PET,
            weight_grams=30.0,
            units_placed=1_000_000,
            recyclability_grade=EPRGrade.C,
        )
        portfolio = _make_portfolio(items=[item])
        result = engine.assess_compliance(portfolio)
        # Weight = 30g * 1M = 30 tonnes, base = 380 EUR/t, mult = 1.0 (grade C)
        # gross = 30 * 380 * 1.0 = 11400
        assert result.total_epr_fee_eur == pytest.approx(11_400.0, rel=0.05)

    def test_total_portfolio_fee(self):
        engine = PackagingComplianceEngine()
        items = [
            _make_item(item_id="A", material=PackagingMaterial.PET, weight_grams=30.0,
                       units_placed=100_000, recyclability_grade=EPRGrade.A),
            _make_item(item_id="B", material=PackagingMaterial.GLASS, weight_grams=200.0,
                       units_placed=50_000, recyclability_grade=EPRGrade.B),
        ]
        portfolio = _make_portfolio(items=items)
        result = engine.assess_compliance(portfolio)
        assert result.total_epr_fee_eur > 0

    def test_modulation_factors(self):
        assert EPR_GRADE_MULTIPLIERS[EPRGrade.A] < EPR_GRADE_MULTIPLIERS[EPRGrade.C]
        assert EPR_GRADE_MULTIPLIERS[EPRGrade.C] < EPR_GRADE_MULTIPLIERS[EPRGrade.E]


# ===================================================================
# TestLabelingCompliance
# ===================================================================


class TestLabelingCompliance:
    """Labeling compliance tests."""

    def test_material_composition_aug2026(self):
        req = LABELING_REQUIREMENTS["material_composition"]
        assert req["deadline"] == "2026-08-01"
        assert req["mandatory"] is True

    def test_sorting_instructions(self):
        req = LABELING_REQUIREMENTS["sorting_instructions"]
        assert req["mandatory"] is True

    def test_pictograms_aug2028(self):
        req = LABELING_REQUIREMENTS["pictograms"]
        assert req["deadline"] == "2028-08-01"

    def test_compliant_item(self):
        engine = PackagingComplianceEngine()
        item = _make_item(
            has_material_marking=True,
            has_sorting_instructions=True,
            has_pictograms=True,
            has_digital_watermark=True,
        )
        portfolio = _make_portfolio(items=[item], reporting_year=2029)
        result = engine.assess_compliance(portfolio)
        # All mandatory labeling should be met
        mandatory_labels = [lc for lc in result.labeling_compliance if lc.mandatory]
        compliant_ones = [lc for lc in mandatory_labels if lc.compliance_pct >= 100.0]
        assert len(compliant_ones) >= 3


# ===================================================================
# TestReuseTargets
# ===================================================================


class TestReuseTargets:
    """Reuse target progress tests."""

    def test_ecommerce_2030_10pct(self):
        targets = PPWR_REUSE_TARGETS["e_commerce"]
        assert targets[2030] == 10.0

    def test_transport_2030_40pct(self):
        targets = PPWR_REUSE_TARGETS["transport_packaging"]
        assert targets[2030] == 40.0

    def test_progress_tracking(self):
        engine = PackagingComplianceEngine()
        items = [
            _make_item(item_id="A", packaging_type=PackagingType.E_COMMERCE,
                       units_placed=10_000, reusable=False),
            _make_item(item_id="B", packaging_type=PackagingType.E_COMMERCE,
                       units_placed=1_000, reusable=True),
        ]
        portfolio = _make_portfolio(items=items)
        result = engine.assess_compliance(portfolio)
        ecom = [r for r in result.reuse_progress if r.packaging_format == "e_commerce"]
        assert len(ecom) == 1
        # 1000 / 11000 ~ 9.09%
        assert ecom[0].reuse_pct == pytest.approx(9.09, abs=1.0)


# ===================================================================
# TestSubstanceRestrictions
# ===================================================================


class TestSubstanceRestrictions:
    """Substance restriction compliance tests."""

    def test_heavy_metals(self):
        assert SUBSTANCE_RESTRICTIONS["heavy_metals"]["limit_ppm"] == 100

    def test_pfas(self):
        assert SUBSTANCE_RESTRICTIONS["pfas"]["limit_ppm"] == 25

    def test_bpa_check(self):
        engine = PackagingComplianceEngine()
        item = _make_item(
            is_contact_sensitive=True,
            contains_bpa=True,
        )
        portfolio = _make_portfolio(items=[item])
        result = engine.assess_compliance(portfolio)
        bpa = [s for s in result.substance_compliance if s.substance == "bisphenol_a"][0]
        assert bpa.compliant is False


# ===================================================================
# TestPackagingCarbon
# ===================================================================


class TestPackagingCarbon:
    """Packaging carbon footprint tests."""

    def test_virgin_vs_recycled(self):
        engine = PackagingComplianceEngine()
        # 100% virgin PET
        item_virgin = _make_item(
            item_id="V", material=PackagingMaterial.PET,
            weight_grams=30.0, units_placed=1_000_000,
            recycled_content_pct=0.0,
        )
        # 50% recycled PET
        item_recycled = _make_item(
            item_id="R", material=PackagingMaterial.PET,
            weight_grams=30.0, units_placed=1_000_000,
            recycled_content_pct=50.0,
        )
        p_v = _make_portfolio(items=[item_virgin])
        p_r = _make_portfolio(items=[item_recycled])
        r_v = engine.assess_compliance(p_v)
        r_r = engine.assess_compliance(p_r)
        assert r_r.carbon_footprint.total_actual_footprint_tco2e < r_v.carbon_footprint.total_actual_footprint_tco2e

    def test_carbon_savings(self):
        engine = PackagingComplianceEngine()
        item = _make_item(
            material=PackagingMaterial.PET,
            weight_grams=30.0, units_placed=1_000_000,
            recycled_content_pct=50.0,
        )
        portfolio = _make_portfolio(items=[item])
        result = engine.assess_compliance(portfolio)
        assert result.carbon_footprint.avoided_emissions_tco2e > 0

    def test_total_footprint(self):
        engine = PackagingComplianceEngine()
        item = _make_item(
            material=PackagingMaterial.PET,
            weight_grams=30.0, units_placed=1_000_000,
            recycled_content_pct=0.0,
        )
        portfolio = _make_portfolio(items=[item])
        result = engine.assess_compliance(portfolio)
        # 30g * 1M = 30 tonnes, virgin PET = 3.14 kgCO2e/kg -> 30000 kg * 3.14 / 1000 = 94.2 tCO2e
        assert result.carbon_footprint.total_virgin_footprint_tco2e == pytest.approx(94.2, rel=1e-2)


# ===================================================================
# TestProvenance
# ===================================================================


class TestProvenance:
    """Provenance hash tests."""

    def test_hash(self):
        engine = PackagingComplianceEngine()
        portfolio = _make_portfolio()
        result = engine.assess_compliance(portfolio)
        assert len(result.provenance_hash) == 64

    def test_deterministic(self):
        engine = PackagingComplianceEngine()
        portfolio = _make_portfolio()
        r1 = engine.assess_compliance(portfolio)
        r2 = engine.assess_compliance(portfolio)
        assert r1.provenance_hash == r2.provenance_hash

    def test_different_input(self):
        engine = PackagingComplianceEngine()
        p1 = _make_portfolio(items=[_make_item(item_id="A")])
        p2 = _make_portfolio(items=[_make_item(item_id="B", weight_grams=50.0)])
        r1 = engine.assess_compliance(p1)
        r2 = engine.assess_compliance(p2)
        assert r1.provenance_hash != r2.provenance_hash


# ===================================================================
# TestEdgeCases
# ===================================================================


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_empty_portfolio_rejected(self):
        with pytest.raises(Exception):
            _make_portfolio(items=[])

    def test_single_item(self):
        engine = PackagingComplianceEngine()
        portfolio = _make_portfolio(items=[_make_item()])
        result = engine.assess_compliance(portfolio)
        assert result.total_items == 1

    def test_large_portfolio(self):
        engine = PackagingComplianceEngine()
        items = [
            _make_item(
                item_id=f"PKG{i:04d}",
                material=PackagingMaterial.PET if i % 3 == 0 else PackagingMaterial.GLASS,
                weight_grams=30.0 + (i % 10),
                units_placed=10_000,
            )
            for i in range(100)
        ]
        portfolio = _make_portfolio(items=items)
        result = engine.assess_compliance(portfolio)
        assert result.total_items == 100
        assert result.total_weight_tonnes > 0

    def test_result_fields(self):
        engine = PackagingComplianceEngine()
        portfolio = _make_portfolio()
        result = engine.assess_compliance(portfolio)
        assert isinstance(result, PPWRComplianceResult)
        assert result.organisation_id == "ORG001"
        assert result.engine_version == "1.0.0"
        assert result.processing_time_ms >= 0
        assert 0 <= result.overall_compliance_score <= 100
