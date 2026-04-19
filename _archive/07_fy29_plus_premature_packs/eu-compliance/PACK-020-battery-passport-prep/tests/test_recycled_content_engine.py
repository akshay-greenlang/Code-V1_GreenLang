# -*- coding: utf-8 -*-
"""
Tests for RecycledContentEngine - PACK-020 Engine 2
=====================================================

Comprehensive tests for recycled content tracking per
EU Battery Regulation Art 8.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-020 Battery Passport Prep
"""

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = PACK_ROOT / "engines"


def _load_module(file_name: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ENGINE_DIR / file_name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_module("recycled_content_engine.py", "pack020_test.engines.recycled_content")

RecycledContentEngine = mod.RecycledContentEngine
RecycledContentInput = mod.RecycledContentInput
RecycledContentResult = mod.RecycledContentResult
MaterialInput = mod.MaterialInput
MaterialResult = mod.MaterialResult
CriticalRawMaterial = mod.CriticalRawMaterial
RecycledContentPhase = mod.RecycledContentPhase
ComplianceStatus = mod.ComplianceStatus
VerificationMethod = mod.VerificationMethod
RECYCLED_CONTENT_TARGETS = mod.RECYCLED_CONTENT_TARGETS
PHASE_EFFECTIVE_DATES = mod.PHASE_EFFECTIVE_DATES
MATERIAL_LABELS = mod.MATERIAL_LABELS


@pytest.fixture
def engine():
    return RecycledContentEngine()


@pytest.fixture
def cobalt_input():
    return MaterialInput(
        material=CriticalRawMaterial.COBALT,
        total_kg=Decimal("12.5"),
        recycled_kg=Decimal("2.5"),
        verification_method=VerificationMethod.MASS_BALANCE,
    )


@pytest.fixture
def lithium_input():
    return MaterialInput(
        material=CriticalRawMaterial.LITHIUM,
        total_kg=Decimal("8.0"),
        recycled_kg=Decimal("0.6"),
    )


@pytest.fixture
def basic_rc_input(cobalt_input, lithium_input):
    return RecycledContentInput(
        battery_id="BAT-RC-001",
        materials=[cobalt_input, lithium_input],
    )


class TestEngineInit:
    def test_init(self):
        engine = RecycledContentEngine()
        assert engine.engine_version == "1.0.0"

    def test_results_empty(self):
        engine = RecycledContentEngine()
        assert engine.get_results() == []


class TestEnums:
    def test_critical_raw_materials(self):
        assert len(CriticalRawMaterial) == 5
        assert CriticalRawMaterial.COBALT.value == "cobalt"

    def test_recycled_content_phases(self):
        assert len(RecycledContentPhase) == 3

    def test_compliance_statuses(self):
        assert len(ComplianceStatus) == 5

    def test_verification_methods(self):
        assert len(VerificationMethod) == 6


class TestConstants:
    def test_targets_2031_cobalt(self):
        t = RECYCLED_CONTENT_TARGETS["minimum_2031"]["cobalt"]
        assert t == Decimal("16")

    def test_targets_2036_lithium(self):
        t = RECYCLED_CONTENT_TARGETS["increased_2036"]["lithium"]
        assert t == Decimal("12")

    def test_targets_2031_lead(self):
        t = RECYCLED_CONTENT_TARGETS["minimum_2031"]["lead"]
        assert t == Decimal("85")

    def test_phase_dates(self):
        assert PHASE_EFFECTIVE_DATES["documentation_2028"] == "2028-08-18"

    def test_all_materials_have_labels(self):
        for mat in CriticalRawMaterial:
            assert mat.value in MATERIAL_LABELS


class TestCalculateRecycledContent:
    def test_basic_calculation(self, engine, basic_rc_input):
        result = engine.calculate_recycled_content(basic_rc_input)
        assert isinstance(result, RecycledContentResult)
        assert result.battery_id == "BAT-RC-001"
        assert result.materials_assessed == 2

    def test_cobalt_percentage(self, engine, basic_rc_input):
        result = engine.calculate_recycled_content(basic_rc_input)
        cobalt_result = [m for m in result.material_results if m.material == CriticalRawMaterial.COBALT][0]
        # 2.5 / 12.5 * 100 = 20%
        assert cobalt_result.recycled_content_pct == Decimal("20.00")

    def test_lithium_percentage(self, engine, basic_rc_input):
        result = engine.calculate_recycled_content(basic_rc_input)
        li_result = [m for m in result.material_results if m.material == CriticalRawMaterial.LITHIUM][0]
        # 0.6 / 8.0 * 100 = 7.5%
        assert li_result.recycled_content_pct == Decimal("7.50")

    def test_overall_recycled_content(self, engine, basic_rc_input):
        result = engine.calculate_recycled_content(basic_rc_input)
        # total recycled = 3.1, total weight = 20.5
        # 3.1 / 20.5 * 100 = 15.12...
        assert result.overall_recycled_content_pct > Decimal("0")

    def test_provenance_hash_present(self, engine, basic_rc_input):
        result = engine.calculate_recycled_content(basic_rc_input)
        assert len(result.provenance_hash) == 64

    def test_processing_time(self, engine, basic_rc_input):
        result = engine.calculate_recycled_content(basic_rc_input)
        assert result.processing_time_ms >= 0.0


class TestTargetCompliance:
    def test_cobalt_meets_2031_target(self, engine):
        """Cobalt at 20% exceeds 16% target for 2031."""
        mat = MaterialInput(
            material=CriticalRawMaterial.COBALT,
            total_kg=Decimal("10"),
            recycled_kg=Decimal("2"),
        )
        inp = RecycledContentInput(battery_id="BAT-T", materials=[mat])
        result = engine.calculate_recycled_content(inp, RecycledContentPhase.MINIMUM_2031)
        cobalt = result.material_results[0]
        assert cobalt.compliance_status == ComplianceStatus.EXCEEDS_TARGET

    def test_cobalt_below_2031_target(self, engine):
        """Cobalt at 10% is below 16% target."""
        mat = MaterialInput(
            material=CriticalRawMaterial.COBALT,
            total_kg=Decimal("10"),
            recycled_kg=Decimal("1"),
        )
        inp = RecycledContentInput(battery_id="BAT-T", materials=[mat])
        result = engine.calculate_recycled_content(inp, RecycledContentPhase.MINIMUM_2031)
        cobalt = result.material_results[0]
        assert cobalt.compliance_status == ComplianceStatus.BELOW_TARGET

    def test_documentation_phase_always_compliant(self, engine):
        mat = MaterialInput(
            material=CriticalRawMaterial.COBALT,
            total_kg=Decimal("10"),
            recycled_kg=Decimal("0.1"),
        )
        inp = RecycledContentInput(battery_id="BAT-T", materials=[mat])
        result = engine.calculate_recycled_content(inp, RecycledContentPhase.DOCUMENTATION_2028)
        cobalt = result.material_results[0]
        assert cobalt.compliance_status == ComplianceStatus.DOCUMENTATION_ONLY

    def test_meets_target_exactly(self, engine):
        mat = MaterialInput(
            material=CriticalRawMaterial.COBALT,
            total_kg=Decimal("100"),
            recycled_kg=Decimal("16"),
        )
        inp = RecycledContentInput(battery_id="BAT-T", materials=[mat])
        result = engine.calculate_recycled_content(inp, RecycledContentPhase.MINIMUM_2031)
        cobalt = result.material_results[0]
        assert cobalt.compliance_status == ComplianceStatus.MEETS_TARGET

    def test_manganese_no_target(self, engine):
        mat = MaterialInput(
            material=CriticalRawMaterial.MANGANESE,
            total_kg=Decimal("5"),
            recycled_kg=Decimal("0"),
        )
        inp = RecycledContentInput(battery_id="BAT-T", materials=[mat])
        result = engine.calculate_recycled_content(inp, RecycledContentPhase.MINIMUM_2031)
        mn = result.material_results[0]
        assert mn.compliance_status == ComplianceStatus.NO_TARGET


class TestGapAnalysis:
    def test_gap_pct_positive_when_below(self, engine):
        mat = MaterialInput(
            material=CriticalRawMaterial.COBALT,
            total_kg=Decimal("100"),
            recycled_kg=Decimal("10"),
        )
        inp = RecycledContentInput(battery_id="BAT-G", materials=[mat])
        result = engine.calculate_recycled_content(inp, RecycledContentPhase.MINIMUM_2031)
        cobalt = result.material_results[0]
        assert cobalt.gap_pct > Decimal("0")  # target is 16, actual is 10 -> gap = 6

    def test_gap_kg_positive_when_below(self, engine):
        mat = MaterialInput(
            material=CriticalRawMaterial.COBALT,
            total_kg=Decimal("100"),
            recycled_kg=Decimal("10"),
        )
        inp = RecycledContentInput(battery_id="BAT-G", materials=[mat])
        result = engine.calculate_recycled_content(inp, RecycledContentPhase.MINIMUM_2031)
        cobalt = result.material_results[0]
        assert cobalt.gap_kg > Decimal("0")

    def test_gap_zero_when_met(self, engine):
        mat = MaterialInput(
            material=CriticalRawMaterial.COBALT,
            total_kg=Decimal("100"),
            recycled_kg=Decimal("20"),
        )
        inp = RecycledContentInput(battery_id="BAT-G", materials=[mat])
        result = engine.calculate_recycled_content(inp, RecycledContentPhase.MINIMUM_2031)
        cobalt = result.material_results[0]
        assert cobalt.gap_kg == Decimal("0.000")


class TestCheckTargets:
    def test_check_targets_compliant(self, engine):
        result = engine.check_targets(
            Decimal("20"), CriticalRawMaterial.COBALT, RecycledContentPhase.MINIMUM_2031
        )
        assert result["compliant"] is True

    def test_check_targets_non_compliant(self, engine):
        result = engine.check_targets(
            Decimal("5"), CriticalRawMaterial.COBALT, RecycledContentPhase.MINIMUM_2031
        )
        assert result["compliant"] is False

    def test_check_targets_has_provenance(self, engine):
        result = engine.check_targets(
            Decimal("20"), CriticalRawMaterial.COBALT, RecycledContentPhase.MINIMUM_2031
        )
        assert len(result["provenance_hash"]) == 64


class TestTargetLookup:
    def test_get_target_cobalt_2031(self, engine):
        t = engine.get_target_for_material(CriticalRawMaterial.COBALT, RecycledContentPhase.MINIMUM_2031)
        assert t == Decimal("16")

    def test_get_all_targets(self, engine):
        targets = engine.get_all_targets(RecycledContentPhase.MINIMUM_2031)
        assert "cobalt" in targets

    def test_get_phase_info(self, engine):
        info = engine.get_phase_info(RecycledContentPhase.MINIMUM_2031)
        assert info["phase"] == "minimum_2031"
        assert "targets" in info

    def test_get_material_info(self, engine):
        info = engine.get_material_info(CriticalRawMaterial.COBALT)
        assert info["material"] == "cobalt"
        assert "targets_by_phase" in info


class TestMultiPhaseAssessment:
    def test_assess_all_phases(self, engine, basic_rc_input):
        result = engine.assess_all_phases(basic_rc_input)
        assert "phase_results" in result
        assert len(result["phase_results"]) == 3
        assert "readiness_summary" in result


class TestBatchProcessing:
    def test_batch_returns_results(self, engine, basic_rc_input):
        results = engine.calculate_batch([basic_rc_input])
        assert len(results) == 1

    def test_batch_empty(self, engine):
        results = engine.calculate_batch([])
        assert results == []


class TestBuildDocumentation:
    def test_documentation_structure(self, engine, basic_rc_input):
        result = engine.calculate_recycled_content(basic_rc_input)
        doc = engine.build_documentation(result)
        assert "document_id" in doc
        assert "materials" in doc
        assert "provenance_hash" in doc
        assert len(doc["materials"]) == 2


class TestCompareResults:
    def test_compare_empty(self, engine):
        result = engine.compare_results([])
        assert result["count"] == 0

    def test_compare_single(self, engine, basic_rc_input):
        r = engine.calculate_recycled_content(basic_rc_input)
        comparison = engine.compare_results([r])
        assert comparison["count"] == 1
        assert "statistics" in comparison


class TestRegistryManagement:
    def test_clear_results(self, engine, basic_rc_input):
        engine.calculate_recycled_content(basic_rc_input)
        assert len(engine.get_results()) == 1
        engine.clear_results()
        assert len(engine.get_results()) == 0


class TestRecommendations:
    def test_non_compliant_generates_recommendations(self, engine):
        mat = MaterialInput(
            material=CriticalRawMaterial.COBALT,
            total_kg=Decimal("100"),
            recycled_kg=Decimal("5"),
        )
        inp = RecycledContentInput(battery_id="BAT-R", materials=[mat])
        result = engine.calculate_recycled_content(inp, RecycledContentPhase.MINIMUM_2031)
        assert len(result.recommendations) > 0

    def test_unverified_generates_recommendation(self, engine):
        mat = MaterialInput(
            material=CriticalRawMaterial.COBALT,
            total_kg=Decimal("100"),
            recycled_kg=Decimal("20"),
            verification_method=VerificationMethod.NOT_VERIFIED,
        )
        inp = RecycledContentInput(battery_id="BAT-R", materials=[mat])
        result = engine.calculate_recycled_content(inp, RecycledContentPhase.MINIMUM_2031)
        has_verify_rec = any("verification" in r.lower() or "not_verified" in r.lower() for r in result.recommendations)
        assert has_verify_rec


class TestInputValidation:
    def test_recycled_exceeds_total_rejected(self):
        with pytest.raises(Exception):
            MaterialInput(
                material=CriticalRawMaterial.COBALT,
                total_kg=Decimal("10"),
                recycled_kg=Decimal("15"),
            )

    def test_duplicate_materials_rejected(self):
        with pytest.raises(Exception):
            RecycledContentInput(
                battery_id="BAT-DUP",
                materials=[
                    MaterialInput(material=CriticalRawMaterial.COBALT, total_kg=Decimal("10"), recycled_kg=Decimal("1")),
                    MaterialInput(material=CriticalRawMaterial.COBALT, total_kg=Decimal("5"), recycled_kg=Decimal("0.5")),
                ],
            )
