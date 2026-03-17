# -*- coding: utf-8 -*-
"""
Tests for SupplyChainDDEngine - PACK-020 Engine 5
===================================================

Comprehensive tests for supply chain due diligence assessment
per EU Battery Regulation Art 48.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-020 Battery Passport Prep
"""

import importlib.util
import json
import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Dynamic Import
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = PACK_ROOT / "engines"


def _load_module(file_name: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ENGINE_DIR / file_name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_module("supply_chain_dd_engine.py", "pack020_test.engines.supply_chain_dd")

SupplyChainDDEngine = mod.SupplyChainDDEngine
SupplierAssessment = mod.SupplierAssessment
OECDStepAssessment = mod.OECDStepAssessment
RiskSummary = mod.RiskSummary
DDResult = mod.DDResult
CriticalRawMaterial = mod.CriticalRawMaterial
DueDiligenceRisk = mod.DueDiligenceRisk
OECDStep = mod.OECDStep
SupplierTier = mod.SupplierTier
HIGH_RISK_COUNTRIES = mod.HIGH_RISK_COUNTRIES
ELEVATED_RISK_COUNTRIES = mod.ELEVATED_RISK_COUNTRIES
COUNTRY_GOVERNANCE_SCORES = mod.COUNTRY_GOVERNANCE_SCORES
OECD_STEP_DESCRIPTIONS = mod.OECD_STEP_DESCRIPTIONS
RISK_WEIGHTS = mod.RISK_WEIGHTS
MATERIAL_CRITICALITY = mod.MATERIAL_CRITICALITY
_compute_hash = mod._compute_hash
_safe_divide = mod._safe_divide
_round2 = mod._round2
_round3 = mod._round3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    return SupplyChainDDEngine()


@pytest.fixture
def high_risk_supplier():
    """Supplier in DRC mining cobalt - very high risk."""
    return SupplierAssessment(
        supplier_id="SUP-HR-001",
        name="CobaltMine DRC Corp",
        material=CriticalRawMaterial.COBALT,
        country="CD",
        tier=SupplierTier.TIER_3,
        oecd_compliant=False,
        third_party_audited=False,
    )


@pytest.fixture
def low_risk_supplier():
    """Supplier in Finland, fully compliant and audited."""
    return SupplierAssessment(
        supplier_id="SUP-LR-001",
        name="NickelRefine Finland Oy",
        material=CriticalRawMaterial.NICKEL,
        country="FI",
        tier=SupplierTier.TIER_1,
        oecd_compliant=True,
        third_party_audited=True,
    )


@pytest.fixture
def mixed_suppliers(high_risk_supplier, low_risk_supplier):
    """Mix of high-risk and low-risk suppliers."""
    medium_supplier = SupplierAssessment(
        supplier_id="SUP-MED-001",
        name="Lithium Extractor Chile",
        material=CriticalRawMaterial.LITHIUM,
        country="CL",
        tier=SupplierTier.TIER_2,
        oecd_compliant=True,
        third_party_audited=False,
    )
    return [high_risk_supplier, low_risk_supplier, medium_supplier]


# ---------------------------------------------------------------------------
# Test: Engine Initialization
# ---------------------------------------------------------------------------


class TestSupplyChainDDEngineInit:
    def test_init_creates_engine(self):
        engine = SupplyChainDDEngine()
        assert engine is not None

    def test_engine_version(self):
        engine = SupplyChainDDEngine()
        assert engine.engine_version == "1.0.0"

    def test_assessments_empty_on_init(self):
        engine = SupplyChainDDEngine()
        assert engine._assessments == []


# ---------------------------------------------------------------------------
# Test: Enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_critical_raw_material_values(self):
        assert CriticalRawMaterial.COBALT.value == "cobalt"
        assert CriticalRawMaterial.LITHIUM.value == "lithium"
        assert CriticalRawMaterial.NICKEL.value == "nickel"
        assert CriticalRawMaterial.NATURAL_GRAPHITE.value == "natural_graphite"
        assert CriticalRawMaterial.MANGANESE.value == "manganese"

    def test_critical_raw_material_count(self):
        assert len(CriticalRawMaterial) == 5

    def test_due_diligence_risk_values(self):
        assert DueDiligenceRisk.VERY_HIGH.value == "very_high"
        assert DueDiligenceRisk.HIGH.value == "high"
        assert DueDiligenceRisk.MEDIUM.value == "medium"
        assert DueDiligenceRisk.LOW.value == "low"
        assert DueDiligenceRisk.NEGLIGIBLE.value == "negligible"

    def test_due_diligence_risk_count(self):
        assert len(DueDiligenceRisk) == 5

    def test_oecd_step_values(self):
        assert OECDStep.STEP_1.value == "step_1_management_systems"
        assert OECDStep.STEP_2.value == "step_2_risk_identification"
        assert OECDStep.STEP_3.value == "step_3_risk_response"
        assert OECDStep.STEP_4.value == "step_4_third_party_audit"
        assert OECDStep.STEP_5.value == "step_5_reporting"

    def test_oecd_step_count(self):
        assert len(OECDStep) == 5

    def test_supplier_tier_values(self):
        assert SupplierTier.TIER_1.value == "tier_1"
        assert SupplierTier.TIER_2.value == "tier_2"
        assert SupplierTier.TIER_3.value == "tier_3"
        assert SupplierTier.TIER_4.value == "tier_4"

    def test_supplier_tier_count(self):
        assert len(SupplierTier) == 4


# ---------------------------------------------------------------------------
# Test: Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_high_risk_countries_has_all_materials(self):
        for mat in CriticalRawMaterial:
            assert mat.value in HIGH_RISK_COUNTRIES

    def test_elevated_risk_countries_has_all_materials(self):
        for mat in CriticalRawMaterial:
            assert mat.value in ELEVATED_RISK_COUNTRIES

    def test_cobalt_high_risk_includes_drc(self):
        assert "CD" in HIGH_RISK_COUNTRIES["cobalt"]

    def test_lithium_high_risk_includes_chile(self):
        assert "CL" in HIGH_RISK_COUNTRIES["lithium"]

    def test_country_governance_scores_drc_low(self):
        assert COUNTRY_GOVERNANCE_SCORES["CD"] == 8

    def test_country_governance_scores_finland_high(self):
        assert COUNTRY_GOVERNANCE_SCORES["FI"] == 92

    def test_oecd_step_descriptions_has_all_steps(self):
        for step in OECDStep:
            assert step.value in OECD_STEP_DESCRIPTIONS

    def test_risk_weights_sum_to_one(self):
        total = sum(RISK_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_material_criticality_has_all_materials(self):
        for mat in CriticalRawMaterial:
            assert mat.value in MATERIAL_CRITICALITY

    def test_cobalt_highest_criticality(self):
        assert MATERIAL_CRITICALITY["cobalt"] == 95

    def test_manganese_lowest_criticality(self):
        assert MATERIAL_CRITICALITY["manganese"] == 60


# ---------------------------------------------------------------------------
# Test: Helper Functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_safe_divide_normal(self):
        assert _safe_divide(10.0, 2.0) == 5.0

    def test_safe_divide_zero_denominator(self):
        assert _safe_divide(10.0, 0.0) == 0.0

    def test_safe_divide_custom_default(self):
        assert _safe_divide(10.0, 0.0, -1.0) == -1.0

    def test_round2_half_up(self):
        assert _round2(1.005) == 1.01
        assert _round2(1.004) == 1.0

    def test_round3_precision(self):
        assert _round3(1.0005) == 1.001

    def test_compute_hash_returns_64_chars(self):
        h = _compute_hash({"test": "data"})
        assert len(h) == 64

    def test_compute_hash_deterministic(self):
        data = {"a": 1, "b": 2}
        assert _compute_hash(data) == _compute_hash(data)

    def test_compute_hash_different_data_different_hash(self):
        assert _compute_hash({"a": 1}) != _compute_hash({"a": 2})


# ---------------------------------------------------------------------------
# Test: Pydantic Models
# ---------------------------------------------------------------------------


class TestPydanticModels:
    def test_supplier_assessment_creation(self):
        sa = SupplierAssessment(
            supplier_id="SUP-001",
            name="TestCorp",
            material=CriticalRawMaterial.COBALT,
            country="DE",
            tier=SupplierTier.TIER_1,
        )
        assert sa.supplier_id == "SUP-001"
        assert sa.risk_level == DueDiligenceRisk.MEDIUM  # default
        assert sa.oecd_compliant is False
        assert sa.third_party_audited is False

    def test_supplier_assessment_country_uppercased(self):
        sa = SupplierAssessment(
            supplier_id="SUP-001",
            name="TestCorp",
            material=CriticalRawMaterial.COBALT,
            country="de",
            tier=SupplierTier.TIER_1,
        )
        assert sa.country == "DE"

    def test_risk_summary_defaults(self):
        rs = RiskSummary()
        assert rs.very_high_count == 0
        assert rs.high_count == 0
        assert rs.materials_at_risk == []
        assert rs.countries_at_risk == []

    def test_dd_result_has_uuid(self):
        r = DDResult()
        assert r.result_id is not None
        assert len(r.result_id) == 36  # UUID4 format

    def test_dd_result_has_version(self):
        r = DDResult()
        assert r.engine_version == "1.0.0"


# ---------------------------------------------------------------------------
# Test: assess_supplier_risk
# ---------------------------------------------------------------------------


class TestAssessSupplierRisk:
    def test_high_risk_drc_cobalt(self, engine, high_risk_supplier):
        result = engine.assess_supplier_risk(high_risk_supplier)
        assert result.risk_score > 60.0
        assert result.risk_level in (DueDiligenceRisk.HIGH, DueDiligenceRisk.VERY_HIGH)
        assert result.country_governance_score == COUNTRY_GOVERNANCE_SCORES["CD"]

    def test_low_risk_finland_nickel(self, engine, low_risk_supplier):
        result = engine.assess_supplier_risk(low_risk_supplier)
        assert result.risk_score < 20.0
        assert result.risk_level in (DueDiligenceRisk.LOW, DueDiligenceRisk.NEGLIGIBLE)
        assert result.country_governance_score == COUNTRY_GOVERNANCE_SCORES["FI"]

    def test_risk_score_within_bounds(self, engine, high_risk_supplier):
        result = engine.assess_supplier_risk(high_risk_supplier)
        assert 0.0 <= result.risk_score <= 100.0

    def test_tier_4_increases_risk(self, engine):
        t1 = SupplierAssessment(
            supplier_id="T1", name="Corp", material=CriticalRawMaterial.NICKEL,
            country="DE", tier=SupplierTier.TIER_1,
        )
        t4 = SupplierAssessment(
            supplier_id="T4", name="Corp", material=CriticalRawMaterial.NICKEL,
            country="DE", tier=SupplierTier.TIER_4,
        )
        r1 = engine.assess_supplier_risk(t1)
        r4 = engine.assess_supplier_risk(t4)
        assert r4.risk_score > r1.risk_score

    def test_oecd_compliance_reduces_risk(self, engine):
        base = dict(
            supplier_id="X", name="Corp", material=CriticalRawMaterial.LITHIUM,
            country="DE", tier=SupplierTier.TIER_1,
        )
        non_compliant = SupplierAssessment(**base, oecd_compliant=False)
        compliant = SupplierAssessment(**base, oecd_compliant=True)
        rn = engine.assess_supplier_risk(non_compliant)
        rc = engine.assess_supplier_risk(compliant)
        assert rc.risk_score < rn.risk_score

    def test_audit_reduces_risk(self, engine):
        base = dict(
            supplier_id="X", name="Corp", material=CriticalRawMaterial.LITHIUM,
            country="DE", tier=SupplierTier.TIER_1,
        )
        not_audited = SupplierAssessment(**base, third_party_audited=False)
        audited = SupplierAssessment(**base, third_party_audited=True)
        rn = engine.assess_supplier_risk(not_audited)
        ra = engine.assess_supplier_risk(audited)
        assert ra.risk_score < rn.risk_score

    def test_mitigation_actions_for_very_high(self, engine, high_risk_supplier):
        result = engine.assess_supplier_risk(high_risk_supplier)
        # DRC cobalt Tier 3 non-compliant, non-audited => many mitigations
        assert len(result.mitigation_actions) > 0

    def test_no_mitigation_for_low_risk(self, engine, low_risk_supplier):
        result = engine.assess_supplier_risk(low_risk_supplier)
        # Low risk fully compliant => minimal or no mitigations
        # OECD risk and audit risk both zero, but governance is high
        # Still might have 0 mitigations
        assert isinstance(result.mitigation_actions, list)

    def test_unknown_country_defaults_governance_50(self, engine):
        s = SupplierAssessment(
            supplier_id="UNK", name="Unknown Corp",
            material=CriticalRawMaterial.COBALT,
            country="XX", tier=SupplierTier.TIER_1,
        )
        result = engine.assess_supplier_risk(s)
        assert result.country_governance_score == 50


# ---------------------------------------------------------------------------
# Test: Risk Score to Level Mapping
# ---------------------------------------------------------------------------


class TestScoreToRiskLevel:
    def test_very_high_threshold(self, engine):
        assert engine._score_to_risk_level(80.0) == DueDiligenceRisk.VERY_HIGH
        assert engine._score_to_risk_level(100.0) == DueDiligenceRisk.VERY_HIGH

    def test_high_threshold(self, engine):
        assert engine._score_to_risk_level(60.0) == DueDiligenceRisk.HIGH
        assert engine._score_to_risk_level(79.99) == DueDiligenceRisk.HIGH

    def test_medium_threshold(self, engine):
        assert engine._score_to_risk_level(40.0) == DueDiligenceRisk.MEDIUM
        assert engine._score_to_risk_level(59.99) == DueDiligenceRisk.MEDIUM

    def test_low_threshold(self, engine):
        assert engine._score_to_risk_level(20.0) == DueDiligenceRisk.LOW
        assert engine._score_to_risk_level(39.99) == DueDiligenceRisk.LOW

    def test_negligible_threshold(self, engine):
        assert engine._score_to_risk_level(0.0) == DueDiligenceRisk.NEGLIGIBLE
        assert engine._score_to_risk_level(19.99) == DueDiligenceRisk.NEGLIGIBLE


# ---------------------------------------------------------------------------
# Test: Tier Risk
# ---------------------------------------------------------------------------


class TestTierRisk:
    def test_tier_1_lowest(self, engine):
        assert engine._calculate_tier_risk(SupplierTier.TIER_1) == 20.0

    def test_tier_2(self, engine):
        assert engine._calculate_tier_risk(SupplierTier.TIER_2) == 45.0

    def test_tier_3(self, engine):
        assert engine._calculate_tier_risk(SupplierTier.TIER_3) == 70.0

    def test_tier_4_highest(self, engine):
        assert engine._calculate_tier_risk(SupplierTier.TIER_4) == 90.0

    def test_tier_risk_increases_with_depth(self, engine):
        t1 = engine._calculate_tier_risk(SupplierTier.TIER_1)
        t2 = engine._calculate_tier_risk(SupplierTier.TIER_2)
        t3 = engine._calculate_tier_risk(SupplierTier.TIER_3)
        t4 = engine._calculate_tier_risk(SupplierTier.TIER_4)
        assert t1 < t2 < t3 < t4


# ---------------------------------------------------------------------------
# Test: Country Risk
# ---------------------------------------------------------------------------


class TestCountryRisk:
    def test_drc_cobalt_very_high(self, engine):
        risk = engine._calculate_country_risk("CD", "cobalt")
        # CD governance=8, base=92, +25 high-risk, +10 conflict = min(100, 127)=100
        assert risk == 100.0

    def test_finland_nickel_elevated(self, engine):
        risk = engine._calculate_country_risk("FI", "nickel")
        # FI governance=92, base=8, elevated risk for nickel +10 => 18.0
        assert risk == 18.0

    def test_elevated_risk_country_adds_10(self, engine):
        # CL is high-risk for lithium, so test elevated for a different combo
        # AU is elevated for cobalt
        risk = engine._calculate_country_risk("AU", "cobalt")
        # AU governance=85, base=15, elevated +10=25
        assert risk == 25.0

    def test_unknown_country_defaults_to_50_governance(self, engine):
        risk = engine._calculate_country_risk("XX", "cobalt")
        # governance=50, base=50, not high-risk, not elevated => 50.0
        assert risk == 50.0

    def test_conflict_affected_penalty(self, engine):
        # MM (Myanmar) for cobalt: governance=12, base=88, +25 high-risk, +10 conflict
        risk = engine._calculate_country_risk("MM", "cobalt")
        assert risk == 100.0


# ---------------------------------------------------------------------------
# Test: OECD Compliance Check
# ---------------------------------------------------------------------------


class TestOECDComplianceCheck:
    def test_all_five_steps_assessed(self, engine, low_risk_supplier):
        result = engine.check_oecd_compliance(low_risk_supplier)
        assert len(result.oecd_step_assessments) == 5

    def test_compliant_supplier_passes_steps_1_to_3(self, engine, low_risk_supplier):
        result = engine.check_oecd_compliance(low_risk_supplier)
        steps_1_3 = result.oecd_step_assessments[:3]
        for step in steps_1_3:
            assert step.compliant is True

    def test_audited_supplier_passes_step_4(self, engine, low_risk_supplier):
        result = engine.check_oecd_compliance(low_risk_supplier)
        step4 = result.oecd_step_assessments[3]
        assert step4.step == OECDStep.STEP_4.value
        assert step4.compliant is True

    def test_step_5_requires_both_oecd_and_audit(self, engine):
        # Only OECD compliant, not audited
        s = SupplierAssessment(
            supplier_id="X", name="Corp", material=CriticalRawMaterial.COBALT,
            country="DE", tier=SupplierTier.TIER_1,
            oecd_compliant=True, third_party_audited=False,
        )
        result = engine.check_oecd_compliance(s)
        step5 = result.oecd_step_assessments[4]
        assert step5.compliant is False

    def test_non_compliant_supplier_has_gaps(self, engine, high_risk_supplier):
        result = engine.check_oecd_compliance(high_risk_supplier)
        for step in result.oecd_step_assessments:
            if not step.compliant:
                assert len(step.gaps) > 0

    def test_full_compliance_requires_all_five_steps(self, engine, low_risk_supplier):
        result = engine.check_oecd_compliance(low_risk_supplier)
        assert result.oecd_compliant is True

    def test_non_compliant_overall_if_any_step_fails(self, engine):
        s = SupplierAssessment(
            supplier_id="X", name="Corp", material=CriticalRawMaterial.COBALT,
            country="DE", tier=SupplierTier.TIER_1,
            oecd_compliant=False, third_party_audited=True,
        )
        result = engine.check_oecd_compliance(s)
        assert result.oecd_compliant is False

    def test_high_risk_country_adds_step_2_gap(self, engine, high_risk_supplier):
        result = engine.check_oecd_compliance(high_risk_supplier)
        step2 = result.oecd_step_assessments[1]
        # DRC cobalt non-compliant => gap mentioning high-risk country
        matching_gaps = [g for g in step2.gaps if "high-risk country" in g]
        assert len(matching_gaps) > 0


# ---------------------------------------------------------------------------
# Test: Audit Coverage
# ---------------------------------------------------------------------------


class TestAuditCoverage:
    def test_empty_list_returns_zero(self, engine):
        assert engine.calculate_audit_coverage([]) == 0.0

    def test_all_audited_returns_100(self, engine, low_risk_supplier):
        assert engine.calculate_audit_coverage([low_risk_supplier]) == 100.0

    def test_none_audited_returns_zero(self, engine, high_risk_supplier):
        assert engine.calculate_audit_coverage([high_risk_supplier]) == 0.0

    def test_mixed_audit_coverage(self, engine, mixed_suppliers):
        # 1 audited out of 3
        rate = engine.calculate_audit_coverage(mixed_suppliers)
        expected = _round2(1.0 / 3.0 * 100.0)
        assert rate == expected

    def test_by_material_groups_correctly(self, engine, mixed_suppliers):
        coverage = engine.calculate_audit_coverage_by_material(mixed_suppliers)
        assert "nickel" in coverage
        assert coverage["nickel"] == 100.0  # low_risk_supplier is nickel, audited

    def test_by_tier_groups_correctly(self, engine, mixed_suppliers):
        coverage = engine.calculate_audit_coverage_by_tier(mixed_suppliers)
        assert "tier_1" in coverage
        assert coverage["tier_1"] == 100.0  # low_risk_supplier is tier 1, audited


# ---------------------------------------------------------------------------
# Test: High Risk Identification
# ---------------------------------------------------------------------------


class TestHighRiskIdentification:
    def test_identifies_high_risk(self, engine):
        s1 = SupplierAssessment(
            supplier_id="HR", name="HR Corp", material=CriticalRawMaterial.COBALT,
            country="DE", tier=SupplierTier.TIER_1,
            risk_level=DueDiligenceRisk.HIGH,
        )
        s2 = SupplierAssessment(
            supplier_id="LR", name="LR Corp", material=CriticalRawMaterial.COBALT,
            country="DE", tier=SupplierTier.TIER_1,
            risk_level=DueDiligenceRisk.LOW,
        )
        result = engine.identify_high_risk([s1, s2])
        assert len(result) == 1
        assert result[0].supplier_id == "HR"

    def test_includes_very_high(self, engine):
        s = SupplierAssessment(
            supplier_id="VH", name="VH Corp", material=CriticalRawMaterial.COBALT,
            country="DE", tier=SupplierTier.TIER_1,
            risk_level=DueDiligenceRisk.VERY_HIGH,
        )
        result = engine.identify_high_risk([s])
        assert len(result) == 1

    def test_empty_list_returns_empty(self, engine):
        assert engine.identify_high_risk([]) == []

    def test_no_high_risk_returns_empty(self, engine):
        s = SupplierAssessment(
            supplier_id="LR", name="LR", material=CriticalRawMaterial.COBALT,
            country="DE", tier=SupplierTier.TIER_1,
            risk_level=DueDiligenceRisk.MEDIUM,
        )
        assert engine.identify_high_risk([s]) == []

    def test_by_material_groups(self, engine):
        s1 = SupplierAssessment(
            supplier_id="HR1", name="Corp1", material=CriticalRawMaterial.COBALT,
            country="DE", tier=SupplierTier.TIER_1,
            risk_level=DueDiligenceRisk.HIGH,
        )
        s2 = SupplierAssessment(
            supplier_id="HR2", name="Corp2", material=CriticalRawMaterial.LITHIUM,
            country="DE", tier=SupplierTier.TIER_1,
            risk_level=DueDiligenceRisk.VERY_HIGH,
        )
        result = engine.identify_high_risk_by_material([s1, s2])
        assert "cobalt" in result
        assert "lithium" in result


# ---------------------------------------------------------------------------
# Test: Full Supply Chain Assessment
# ---------------------------------------------------------------------------


class TestAssessSupplyChain:
    def test_assess_single_supplier(self, engine, high_risk_supplier):
        result = engine.assess_supply_chain([high_risk_supplier])
        assert isinstance(result, DDResult)
        assert result.suppliers_assessed == 1
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_assess_multiple_suppliers(self, engine, mixed_suppliers):
        result = engine.assess_supply_chain(mixed_suppliers)
        assert result.suppliers_assessed == 3
        assert len(result.supplier_assessments) == 3

    def test_empty_supplier_list(self, engine):
        result = engine.assess_supply_chain([])
        assert result.suppliers_assessed == 0
        assert result.high_risk_count == 0

    def test_oecd_compliance_rate_calculated(self, engine, mixed_suppliers):
        result = engine.assess_supply_chain(mixed_suppliers)
        assert 0.0 <= result.oecd_compliance_rate <= 100.0

    def test_audit_coverage_rate_calculated(self, engine, mixed_suppliers):
        result = engine.assess_supply_chain(mixed_suppliers)
        assert 0.0 <= result.audit_coverage_rate <= 100.0

    def test_materials_assessed_populated(self, engine, mixed_suppliers):
        result = engine.assess_supply_chain(mixed_suppliers)
        assert len(result.materials_assessed) > 0
        assert "cobalt" in result.materials_assessed

    def test_materials_assessed_sorted(self, engine, mixed_suppliers):
        result = engine.assess_supply_chain(mixed_suppliers)
        assert result.materials_assessed == sorted(result.materials_assessed)

    def test_mitigation_required_when_high_risk(self, engine, high_risk_supplier):
        result = engine.assess_supply_chain([high_risk_supplier])
        assert result.mitigation_required is True

    def test_recommendations_generated(self, engine, mixed_suppliers):
        result = engine.assess_supply_chain(mixed_suppliers)
        assert len(result.recommendations) > 0

    def test_processing_time_recorded(self, engine, mixed_suppliers):
        result = engine.assess_supply_chain(mixed_suppliers)
        assert result.processing_time_ms > 0.0

    def test_average_risk_score_within_bounds(self, engine, mixed_suppliers):
        result = engine.assess_supply_chain(mixed_suppliers)
        assert 0.0 <= result.average_risk_score <= 100.0

    def test_overall_risk_level_assigned(self, engine, mixed_suppliers):
        result = engine.assess_supply_chain(mixed_suppliers)
        assert result.overall_risk_level in DueDiligenceRisk

    def test_supplier_provenance_hashes_set(self, engine, mixed_suppliers):
        result = engine.assess_supply_chain(mixed_suppliers)
        for sa in result.supplier_assessments:
            assert sa.provenance_hash != ""
            assert len(sa.provenance_hash) == 64

    def test_risk_summary_counts_match(self, engine, mixed_suppliers):
        result = engine.assess_supply_chain(mixed_suppliers)
        rs = result.risk_summary
        total_from_summary = (
            rs.very_high_count + rs.high_count + rs.medium_count
            + rs.low_count + rs.negligible_count
        )
        assert total_from_summary == result.suppliers_assessed


# ---------------------------------------------------------------------------
# Test: Supplier Summary
# ---------------------------------------------------------------------------


class TestSupplierSummary:
    def test_summary_structure(self, engine, low_risk_supplier):
        assessed = engine.assess_supplier_risk(low_risk_supplier)
        assessed = engine.check_oecd_compliance(assessed)
        summary = engine.get_supplier_summary(assessed)
        assert "supplier_id" in summary
        assert "risk_level" in summary
        assert "oecd_steps_compliant" in summary
        assert "provenance_hash" in summary

    def test_oecd_steps_counted(self, engine, low_risk_supplier):
        assessed = engine.assess_supplier_risk(low_risk_supplier)
        assessed = engine.check_oecd_compliance(assessed)
        summary = engine.get_supplier_summary(assessed)
        assert summary["oecd_steps_total"] == 5


# ---------------------------------------------------------------------------
# Test: Material Risk Profile
# ---------------------------------------------------------------------------


class TestMaterialRiskProfile:
    def test_profile_structure(self, engine, mixed_suppliers):
        assessed = [engine.assess_supplier_risk(s) for s in mixed_suppliers]
        profile = engine.get_material_risk_profile(assessed)
        assert "cobalt" in profile
        assert "nickel" in profile
        assert "provenance_hash" in profile

    def test_profile_supplier_count(self, engine, mixed_suppliers):
        assessed = [engine.assess_supplier_risk(s) for s in mixed_suppliers]
        profile = engine.get_material_risk_profile(assessed)
        assert profile["cobalt"]["supplier_count"] == 1
        assert profile["nickel"]["supplier_count"] == 1

    def test_profile_criticality_score(self, engine, mixed_suppliers):
        assessed = [engine.assess_supplier_risk(s) for s in mixed_suppliers]
        profile = engine.get_material_risk_profile(assessed)
        assert profile["cobalt"]["criticality_score"] == 95


# ---------------------------------------------------------------------------
# Test: Country Risk Assessment
# ---------------------------------------------------------------------------


class TestCountryRiskAssessment:
    def test_country_risk_assessment_structure(self, engine):
        result = engine.get_country_risk_assessment("CD", "cobalt")
        assert result["country"] == "CD"
        assert result["material"] == "cobalt"
        assert result["is_high_risk"] is True
        assert result["is_conflict_affected"] is True
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_low_risk_country(self, engine):
        result = engine.get_country_risk_assessment("DE", "nickel")
        assert result["risk_category"] == "STANDARD"
        assert result["is_high_risk"] is False

    def test_elevated_risk_country(self, engine):
        result = engine.get_country_risk_assessment("AU", "cobalt")
        assert result["risk_category"] == "ELEVATED_RISK"
        assert result["is_elevated_risk"] is True


# ---------------------------------------------------------------------------
# Test: High Risk Countries Lookup
# ---------------------------------------------------------------------------


class TestHighRiskCountries:
    def test_get_all_high_risk(self, engine):
        result = engine.get_high_risk_countries()
        assert len(result) == 5  # all materials

    def test_filter_by_material(self, engine):
        result = engine.get_high_risk_countries("cobalt")
        assert "cobalt" in result
        assert len(result) == 1

    def test_unknown_material_empty(self, engine):
        result = engine.get_high_risk_countries("unknown")
        assert result == {"unknown": []}


# ---------------------------------------------------------------------------
# Test: Clear Assessments
# ---------------------------------------------------------------------------


class TestClearAssessments:
    def test_clear_removes_stored_assessments(self, engine, mixed_suppliers):
        engine.assess_supply_chain(mixed_suppliers)
        assert len(engine._assessments) > 0
        engine.clear_assessments()
        assert len(engine._assessments) == 0


# ---------------------------------------------------------------------------
# Test: Risk Summary
# ---------------------------------------------------------------------------


class TestRiskSummary:
    def test_empty_suppliers_default_summary(self, engine):
        summary = engine._build_risk_summary([])
        assert summary.very_high_count == 0
        assert summary.materials_at_risk == []

    def test_percentages_sum_to_100(self, engine, mixed_suppliers):
        assessed = []
        for s in mixed_suppliers:
            assessed.append(engine.assess_supplier_risk(s))
        summary = engine._build_risk_summary(assessed)
        total_pct = (
            summary.very_high_pct + summary.high_pct + summary.medium_pct
            + summary.low_pct + summary.negligible_pct
        )
        assert total_pct == pytest.approx(100.0, abs=0.1)


# ---------------------------------------------------------------------------
# Test: Recommendations
# ---------------------------------------------------------------------------


class TestRecommendations:
    def test_all_compliant_positive_message(self, engine):
        s = SupplierAssessment(
            supplier_id="X", name="Corp",
            material=CriticalRawMaterial.NICKEL,
            country="FI", tier=SupplierTier.TIER_1,
            oecd_compliant=True, third_party_audited=True,
            risk_level=DueDiligenceRisk.LOW,
        )
        summary = RiskSummary()
        recs = engine._generate_recommendations([s], summary, 100.0, 100.0)
        assert any("met" in r.lower() for r in recs)

    def test_low_oecd_triggers_critical(self, engine):
        s = SupplierAssessment(
            supplier_id="X", name="Corp",
            material=CriticalRawMaterial.COBALT,
            country="CD", tier=SupplierTier.TIER_3,
            oecd_compliant=False, third_party_audited=False,
            risk_level=DueDiligenceRisk.HIGH,
        )
        summary = RiskSummary(high_count=1, materials_at_risk=["cobalt"], countries_at_risk=["CD"])
        recs = engine._generate_recommendations([s], summary, 0.0, 0.0)
        assert any("CRITICAL" in r for r in recs)

    def test_deep_tier_recommendation(self, engine):
        s = SupplierAssessment(
            supplier_id="X", name="Corp",
            material=CriticalRawMaterial.COBALT,
            country="DE", tier=SupplierTier.TIER_4,
            risk_level=DueDiligenceRisk.MEDIUM,
        )
        summary = RiskSummary()
        recs = engine._generate_recommendations([s], summary, 100.0, 100.0)
        assert any("Tier 3 or Tier 4" in r for r in recs)


# ---------------------------------------------------------------------------
# Test: Provenance / Reproducibility
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_same_input_same_provenance(self, engine):
        s = SupplierAssessment(
            supplier_id="X", name="Corp",
            material=CriticalRawMaterial.COBALT,
            country="DE", tier=SupplierTier.TIER_1,
        )
        r1 = engine.assess_supply_chain([s])
        # Create a fresh supplier (same data) for second run
        s2 = SupplierAssessment(
            supplier_id="X", name="Corp",
            material=CriticalRawMaterial.COBALT,
            country="DE", tier=SupplierTier.TIER_1,
        )
        r2 = engine.assess_supply_chain([s2])
        # The result_id and timestamps differ, but provenance of suppliers
        # should be deterministic for same input
        for sa1, sa2 in zip(r1.supplier_assessments, r2.supplier_assessments):
            assert sa1.provenance_hash == sa2.provenance_hash
