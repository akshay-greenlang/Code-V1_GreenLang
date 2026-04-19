# -*- coding: utf-8 -*-
"""
Tests for ConformityAssessmentEngine - PACK-020 Engine 8
=========================================================

Comprehensive tests for conformity assessment readiness
per EU Battery Regulation Art 17-22.

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


mod = _load_module("conformity_assessment_engine.py", "pack020_test.engines.conformity_assessment")

ConformityAssessmentEngine = mod.ConformityAssessmentEngine
ConformityInput = mod.ConformityInput
ConformityResult = mod.ConformityResult
DocumentationItem = mod.DocumentationItem
TestResult = mod.TestResult
BatteryCategory = mod.BatteryCategory
ConformityModule = mod.ConformityModule
DocumentationType = mod.DocumentationType
ConformityStatus = mod.ConformityStatus
MODULE_DESCRIPTIONS = mod.MODULE_DESCRIPTIONS
DEFAULT_MODULE_BY_CATEGORY = mod.DEFAULT_MODULE_BY_CATEGORY
NOTIFIED_BODY_MODULES = mod.NOTIFIED_BODY_MODULES
TECHNICAL_DOC_CHECKLIST = mod.TECHNICAL_DOC_CHECKLIST
EU_DOC_CHECKLIST = mod.EU_DOC_CHECKLIST
TEST_REQUIREMENTS = mod.TEST_REQUIREMENTS
_compute_hash = mod._compute_hash
_safe_divide = mod._safe_divide
_round2 = mod._round2
_round3 = mod._round3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    return ConformityAssessmentEngine()


@pytest.fixture
def all_docs_available():
    """All 12 technical documentation items available."""
    return {item_id: True for item_id in TECHNICAL_DOC_CHECKLIST}


@pytest.fixture
def no_docs_available():
    """No technical documentation items available."""
    return {item_id: False for item_id in TECHNICAL_DOC_CHECKLIST}


@pytest.fixture
def all_declaration_available():
    """All 8 EU Declaration items available."""
    return {item_id: True for item_id in EU_DOC_CHECKLIST}


@pytest.fixture
def ev_tests_all_done():
    """All EV tests completed and passed."""
    test_reports = {}
    test_results = {}
    for test_def in TEST_REQUIREMENTS.get("ev", []):
        test_reports[test_def["test"]] = True
        test_results[test_def["test"]] = True
    return test_reports, test_results


@pytest.fixture
def fully_ready_ev_input(all_docs_available, all_declaration_available, ev_tests_all_done):
    """Fully ready EV battery conformity input."""
    test_reports, test_results = ev_tests_all_done
    return ConformityInput(
        battery_id="BAT-READY-001",
        category=BatteryCategory.EV,
        technical_documentation=all_docs_available,
        test_reports=test_reports,
        test_results=test_results,
        eu_declaration=all_declaration_available,
        ce_marking=True,
    )


@pytest.fixture
def empty_ev_input():
    """EV battery with nothing ready."""
    return ConformityInput(
        battery_id="BAT-EMPTY-001",
        category=BatteryCategory.EV,
    )


# ---------------------------------------------------------------------------
# Test: Engine Initialization
# ---------------------------------------------------------------------------


class TestConformityAssessmentEngineInit:
    def test_init_creates_engine(self):
        engine = ConformityAssessmentEngine()
        assert engine is not None

    def test_engine_version(self):
        engine = ConformityAssessmentEngine()
        assert engine.engine_version == "1.0.0"


# ---------------------------------------------------------------------------
# Test: Enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_battery_category_values(self):
        assert BatteryCategory.PORTABLE.value == "portable"
        assert BatteryCategory.LMT.value == "lmt"
        assert BatteryCategory.SLI.value == "sli"
        assert BatteryCategory.EV.value == "ev"
        assert BatteryCategory.INDUSTRIAL.value == "industrial"

    def test_battery_category_count(self):
        assert len(BatteryCategory) == 5

    def test_conformity_module_values(self):
        assert ConformityModule.MODULE_A.value == "module_a"
        assert ConformityModule.MODULE_B.value == "module_b"
        assert ConformityModule.MODULE_C.value == "module_c"
        assert ConformityModule.MODULE_D.value == "module_d"
        assert ConformityModule.MODULE_E.value == "module_e"
        assert ConformityModule.MODULE_G.value == "module_g"
        assert ConformityModule.MODULE_H.value == "module_h"

    def test_conformity_module_count(self):
        assert len(ConformityModule) == 7

    def test_documentation_type_values(self):
        assert DocumentationType.TECHNICAL_FILE.value == "technical_file"
        assert DocumentationType.EU_DECLARATION.value == "eu_declaration"
        assert DocumentationType.TEST_REPORTS.value == "test_reports"
        assert DocumentationType.QUALITY_SYSTEM.value == "quality_system"
        assert DocumentationType.DESIGN_EXAMINATION.value == "design_examination"

    def test_documentation_type_count(self):
        assert len(DocumentationType) == 5

    def test_conformity_status_values(self):
        assert ConformityStatus.READY.value == "ready"
        assert ConformityStatus.PARTIALLY_READY.value == "partially_ready"
        assert ConformityStatus.NOT_READY.value == "not_ready"
        assert ConformityStatus.IN_PROGRESS.value == "in_progress"

    def test_conformity_status_count(self):
        assert len(ConformityStatus) == 4


# ---------------------------------------------------------------------------
# Test: Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_module_descriptions_all_modules(self):
        for module in ConformityModule:
            assert module.value in MODULE_DESCRIPTIONS

    def test_default_module_all_categories(self):
        for cat in BatteryCategory:
            assert cat.value in DEFAULT_MODULE_BY_CATEGORY

    def test_all_defaults_are_module_a(self):
        for cat in BatteryCategory:
            assert DEFAULT_MODULE_BY_CATEGORY[cat.value] == ConformityModule.MODULE_A

    def test_notified_body_modules_list(self):
        assert "module_b" in NOTIFIED_BODY_MODULES
        assert "module_d" in NOTIFIED_BODY_MODULES
        assert "module_e" in NOTIFIED_BODY_MODULES
        assert "module_g" in NOTIFIED_BODY_MODULES
        assert "module_h" in NOTIFIED_BODY_MODULES
        assert "module_a" not in NOTIFIED_BODY_MODULES

    def test_technical_doc_checklist_count(self):
        assert len(TECHNICAL_DOC_CHECKLIST) == 12

    def test_eu_doc_checklist_count(self):
        assert len(EU_DOC_CHECKLIST) == 8

    def test_test_requirements_all_categories(self):
        for cat in BatteryCategory:
            assert cat.value in TEST_REQUIREMENTS

    def test_ev_has_most_tests(self):
        ev_count = len(TEST_REQUIREMENTS["ev"])
        for cat in BatteryCategory:
            assert ev_count >= len(TEST_REQUIREMENTS[cat.value])

    def test_technical_doc_items_have_required_fields(self):
        for item_id, item_def in TECHNICAL_DOC_CHECKLIST.items():
            assert "item" in item_def
            assert "description" in item_def
            assert "article" in item_def

    def test_eu_doc_items_have_required_fields(self):
        for item_id, item_def in EU_DOC_CHECKLIST.items():
            assert "item" in item_def
            assert "description" in item_def
            assert "article" in item_def


# ---------------------------------------------------------------------------
# Test: Module Determination
# ---------------------------------------------------------------------------


class TestModuleDetermination:
    def test_default_module_ev(self, engine):
        module = engine.determine_module(BatteryCategory.EV)
        assert module == ConformityModule.MODULE_A

    def test_default_module_portable(self, engine):
        module = engine.determine_module(BatteryCategory.PORTABLE)
        assert module == ConformityModule.MODULE_A

    def test_explicit_module_override(self, engine):
        module = engine.determine_module(
            BatteryCategory.EV, ConformityModule.MODULE_H
        )
        assert module == ConformityModule.MODULE_H

    def test_explicit_module_b(self, engine):
        module = engine.determine_module(
            BatteryCategory.INDUSTRIAL, ConformityModule.MODULE_B
        )
        assert module == ConformityModule.MODULE_B

    def test_none_selection_uses_default(self, engine):
        module = engine.determine_module(BatteryCategory.SLI, None)
        assert module == ConformityModule.MODULE_A


# ---------------------------------------------------------------------------
# Test: Documentation Completeness
# ---------------------------------------------------------------------------


class TestDocumentationCompleteness:
    def test_all_docs_100_pct(self, engine, all_docs_available):
        items, completeness = engine.check_documentation(
            all_docs_available, BatteryCategory.EV
        )
        assert completeness == 100.0
        assert len(items) == 12

    def test_no_docs_0_pct(self, engine, no_docs_available):
        items, completeness = engine.check_documentation(
            no_docs_available, BatteryCategory.EV
        )
        assert completeness == 0.0

    def test_partial_docs(self, engine):
        docs = {item_id: False for item_id in TECHNICAL_DOC_CHECKLIST}
        docs["td_01"] = True
        docs["td_02"] = True
        items, completeness = engine.check_documentation(
            docs, BatteryCategory.EV
        )
        assert 0.0 < completeness < 100.0

    def test_carbon_footprint_not_required_for_portable(self, engine):
        # Missing td_08 (carbon footprint) should not count against portable
        docs = {item_id: True for item_id in TECHNICAL_DOC_CHECKLIST}
        docs["td_08"] = False
        items, completeness = engine.check_documentation(
            docs, BatteryCategory.PORTABLE
        )
        assert completeness == 100.0

    def test_carbon_footprint_required_for_ev(self, engine):
        docs = {item_id: True for item_id in TECHNICAL_DOC_CHECKLIST}
        docs["td_08"] = False
        items, completeness = engine.check_documentation(
            docs, BatteryCategory.EV
        )
        assert completeness < 100.0

    def test_empty_dict_0_pct(self, engine):
        items, completeness = engine.check_documentation(
            {}, BatteryCategory.EV
        )
        assert completeness == 0.0

    def test_items_have_correct_fields(self, engine, all_docs_available):
        items, _ = engine.check_documentation(
            all_docs_available, BatteryCategory.EV
        )
        for item in items:
            assert item.item_id != ""
            assert item.item_name != ""
            assert item.article != ""

    def test_missing_items_have_notes(self, engine, no_docs_available):
        items, _ = engine.check_documentation(
            no_docs_available, BatteryCategory.EV
        )
        for item in items:
            if not item.available:
                assert item.notes != ""


# ---------------------------------------------------------------------------
# Test: Test Coverage
# ---------------------------------------------------------------------------


class TestTestCoverage:
    def test_all_tests_done_100_pct(self, engine, ev_tests_all_done):
        test_reports, test_results = ev_tests_all_done
        items, coverage, pass_rate = engine.check_test_coverage(
            BatteryCategory.EV, test_reports, test_results
        )
        assert coverage == 100.0
        assert pass_rate == 100.0

    def test_no_tests_done_0_pct(self, engine):
        items, coverage, pass_rate = engine.check_test_coverage(
            BatteryCategory.EV, {}, {}
        )
        assert coverage == 0.0
        assert pass_rate == 0.0

    def test_partial_completion(self, engine):
        ev_tests = TEST_REQUIREMENTS["ev"]
        test_reports = {ev_tests[0]["test"]: True}
        test_results = {ev_tests[0]["test"]: True}
        items, coverage, pass_rate = engine.check_test_coverage(
            BatteryCategory.EV, test_reports, test_results
        )
        assert 0.0 < coverage < 100.0
        assert pass_rate == 100.0

    def test_completed_but_failed(self, engine):
        ev_tests = TEST_REQUIREMENTS["ev"]
        test_reports = {t["test"]: True for t in ev_tests}
        test_results = {t["test"]: False for t in ev_tests}
        items, coverage, pass_rate = engine.check_test_coverage(
            BatteryCategory.EV, test_reports, test_results
        )
        assert coverage == 100.0
        assert pass_rate == 0.0

    def test_ev_has_9_tests(self, engine):
        items, _, _ = engine.check_test_coverage(BatteryCategory.EV, {}, {})
        assert len(items) == 9

    def test_portable_has_4_tests(self, engine):
        items, _, _ = engine.check_test_coverage(BatteryCategory.PORTABLE, {}, {})
        assert len(items) == 4

    def test_conditional_tests_not_counted_required(self, engine):
        # Industrial has a conditional test (grid connection)
        items, _, _ = engine.check_test_coverage(BatteryCategory.INDUSTRIAL, {}, {})
        conditional = [i for i in items if not i.required]
        assert len(conditional) == 1
        assert conditional[0].test_name == "Grid connection"

    def test_report_reference_set_when_completed(self, engine, ev_tests_all_done):
        test_reports, test_results = ev_tests_all_done
        items, _, _ = engine.check_test_coverage(
            BatteryCategory.EV, test_reports, test_results
        )
        for item in items:
            if item.completed:
                assert item.report_reference != ""


# ---------------------------------------------------------------------------
# Test: Declaration Validation
# ---------------------------------------------------------------------------


class TestDeclarationValidation:
    def test_all_items_valid(self, engine, all_declaration_available):
        items, completeness, is_valid = engine.validate_declaration(
            all_declaration_available, ConformityModule.MODULE_A
        )
        assert completeness == 100.0
        assert is_valid is True

    def test_no_items_invalid(self, engine):
        items, completeness, is_valid = engine.validate_declaration(
            {}, ConformityModule.MODULE_A
        )
        assert completeness == 0.0
        assert is_valid is False

    def test_module_a_doc07_not_required(self, engine):
        # For Module A, notified body details (doc_07) not required
        decl = {item_id: True for item_id in EU_DOC_CHECKLIST}
        decl["doc_07"] = False
        items, completeness, is_valid = engine.validate_declaration(
            decl, ConformityModule.MODULE_A
        )
        assert completeness == 100.0
        assert is_valid is True

    def test_module_b_doc07_required(self, engine):
        # For Module B (notified body module), doc_07 required
        decl = {item_id: True for item_id in EU_DOC_CHECKLIST}
        decl["doc_07"] = False
        items, completeness, is_valid = engine.validate_declaration(
            decl, ConformityModule.MODULE_B
        )
        assert completeness < 100.0
        assert is_valid is False

    def test_module_b_doc07_requires_nb_id(self, engine):
        # Module B with doc_07 True but no NB ID => should still be False
        decl = {item_id: True for item_id in EU_DOC_CHECKLIST}
        items, completeness, is_valid = engine.validate_declaration(
            decl, ConformityModule.MODULE_B, notified_body_id=None
        )
        # doc_07 available in dict but NB ID missing => marked False
        doc07 = next(i for i in items if i.item_id == "doc_07")
        assert doc07.available is False

    def test_module_b_doc07_with_nb_id(self, engine):
        decl = {item_id: True for item_id in EU_DOC_CHECKLIST}
        items, completeness, is_valid = engine.validate_declaration(
            decl, ConformityModule.MODULE_B, notified_body_id="NB-1234"
        )
        assert is_valid is True

    def test_items_count(self, engine, all_declaration_available):
        items, _, _ = engine.validate_declaration(
            all_declaration_available, ConformityModule.MODULE_A
        )
        assert len(items) == 8


# ---------------------------------------------------------------------------
# Test: Quality System Check
# ---------------------------------------------------------------------------


class TestQualitySystem:
    def test_module_a_always_adequate(self, engine):
        assert engine._check_quality_system(
            ConformityModule.MODULE_A, False, ""
        ) is True

    def test_module_d_requires_qs(self, engine):
        assert engine._check_quality_system(
            ConformityModule.MODULE_D, False, ""
        ) is False

    def test_module_d_iso9001_adequate(self, engine):
        assert engine._check_quality_system(
            ConformityModule.MODULE_D, True, "ISO 9001:2015"
        ) is True

    def test_module_e_iatf_adequate(self, engine):
        assert engine._check_quality_system(
            ConformityModule.MODULE_E, True, "IATF 16949"
        ) is True

    def test_module_h_requires_qs(self, engine):
        assert engine._check_quality_system(
            ConformityModule.MODULE_H, False, ""
        ) is False

    def test_unrecognized_standard_not_adequate(self, engine):
        assert engine._check_quality_system(
            ConformityModule.MODULE_D, True, "XYZ Standard"
        ) is False

    def test_certified_but_no_standard_not_adequate(self, engine):
        assert engine._check_quality_system(
            ConformityModule.MODULE_D, True, ""
        ) is False

    def test_module_b_not_qs_required(self, engine):
        assert engine._check_quality_system(
            ConformityModule.MODULE_B, False, ""
        ) is True

    def test_module_g_not_qs_required(self, engine):
        assert engine._check_quality_system(
            ConformityModule.MODULE_G, False, ""
        ) is True


# ---------------------------------------------------------------------------
# Test: Status Determination
# ---------------------------------------------------------------------------


class TestStatusDetermination:
    def test_ready_at_100_no_missing(self, engine):
        assert engine._determine_status(100.0, []) == ConformityStatus.READY

    def test_partially_ready_at_70(self, engine):
        assert engine._determine_status(70.0, ["missing"]) == ConformityStatus.PARTIALLY_READY

    def test_partially_ready_at_99_with_missing(self, engine):
        assert engine._determine_status(99.0, ["missing"]) == ConformityStatus.PARTIALLY_READY

    def test_in_progress_at_30(self, engine):
        assert engine._determine_status(30.0, ["missing"]) == ConformityStatus.IN_PROGRESS

    def test_not_ready_below_30(self, engine):
        assert engine._determine_status(29.0, ["missing"]) == ConformityStatus.NOT_READY

    def test_not_ready_at_0(self, engine):
        assert engine._determine_status(0.0, ["missing"]) == ConformityStatus.NOT_READY

    def test_boundary_100_with_missing_is_partially(self, engine):
        assert engine._determine_status(100.0, ["something"]) == ConformityStatus.PARTIALLY_READY


# ---------------------------------------------------------------------------
# Test: Overall Score Calculation
# ---------------------------------------------------------------------------


class TestOverallScore:
    def test_perfect_score(self, engine):
        score = engine._calculate_overall_score(
            100.0, 100.0, 100.0, True, False, False, True
        )
        assert score == 100.0

    def test_zero_score(self, engine):
        score = engine._calculate_overall_score(
            0.0, 0.0, 0.0, False, False, False, True
        )
        # NB not required => nb_score=100, 0*0.3+0*0.3+0*0.2+0*0.1+100*0.1=10
        assert score == 10.0

    def test_zero_score_with_nb_required(self, engine):
        score = engine._calculate_overall_score(
            0.0, 0.0, 0.0, False, True, False, False
        )
        # nb_score=0, qs_not_adequate but nb_required => nb*0.5=0
        assert score == 0.0

    def test_ce_marking_adds_10_pct(self, engine):
        without_ce = engine._calculate_overall_score(
            100.0, 100.0, 100.0, False, False, False, True
        )
        with_ce = engine._calculate_overall_score(
            100.0, 100.0, 100.0, True, False, False, True
        )
        assert with_ce - without_ce == pytest.approx(10.0, abs=0.01)

    def test_nb_required_not_engaged(self, engine):
        score = engine._calculate_overall_score(
            100.0, 100.0, 100.0, True, True, False, True
        )
        # nb_score=0, qs_adequate but nb_required and nb not engaged => 0*0.5=0
        # Actually qs_adequate=True, nb_required=True, nb_engaged=False
        # nb_score = 0 (not engaged); qs_adequate but nb_required => nb_score*0.5=0
        # Wait: "if not qs_adequate and nb_required: nb_score = nb_score * 0.5"
        # qs_adequate=True, so no reduction. nb_score=0 (not engaged).
        assert score < 100.0

    def test_weights_sum_correctly(self, engine):
        # doc=50, test=50, decl=50, ce=True, nb not required
        score = engine._calculate_overall_score(
            50.0, 50.0, 50.0, True, False, False, True
        )
        # 50*0.3 + 50*0.3 + 50*0.2 + 100*0.1 + 100*0.1 = 15+15+10+10+10 = 60
        assert score == 60.0


# ---------------------------------------------------------------------------
# Test: Full Conformity Assessment
# ---------------------------------------------------------------------------


class TestAssessConformity:
    def test_fully_ready(self, engine, fully_ready_ev_input):
        result = engine.assess_conformity(fully_ready_ev_input)
        assert isinstance(result, ConformityResult)
        assert result.overall_status == ConformityStatus.READY
        assert result.overall_score == 100.0
        assert result.documentation_completeness == 100.0
        assert result.test_coverage == 100.0
        assert result.test_pass_rate == 100.0
        assert result.declaration_valid is True
        assert result.ce_marking_applied is True
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_nothing_ready(self, engine, empty_ev_input):
        result = engine.assess_conformity(empty_ev_input)
        assert result.overall_status == ConformityStatus.NOT_READY
        assert result.overall_score < 30.0
        assert result.documentation_completeness == 0.0
        assert result.test_coverage == 0.0
        assert result.ce_marking_applied is False

    def test_battery_id_in_result(self, engine, fully_ready_ev_input):
        result = engine.assess_conformity(fully_ready_ev_input)
        assert result.battery_id == "BAT-READY-001"

    def test_category_in_result(self, engine, fully_ready_ev_input):
        result = engine.assess_conformity(fully_ready_ev_input)
        assert result.category == BatteryCategory.EV

    def test_module_in_result(self, engine, fully_ready_ev_input):
        result = engine.assess_conformity(fully_ready_ev_input)
        assert result.module == ConformityModule.MODULE_A

    def test_module_description_in_result(self, engine, fully_ready_ev_input):
        result = engine.assess_conformity(fully_ready_ev_input)
        assert result.module_description != ""

    def test_missing_items_populated(self, engine, empty_ev_input):
        result = engine.assess_conformity(empty_ev_input)
        assert len(result.missing_items) > 0

    def test_recommendations_populated(self, engine, empty_ev_input):
        result = engine.assess_conformity(empty_ev_input)
        assert len(result.recommendations) > 0

    def test_processing_time(self, engine, fully_ready_ev_input):
        result = engine.assess_conformity(fully_ready_ev_input)
        assert result.processing_time_ms >= 0.0

    def test_notified_body_not_required_module_a(self, engine, fully_ready_ev_input):
        result = engine.assess_conformity(fully_ready_ev_input)
        assert result.notified_body_required is False

    def test_notified_body_required_module_b(self, engine, all_docs_available, all_declaration_available, ev_tests_all_done):
        test_reports, test_results = ev_tests_all_done
        input_data = ConformityInput(
            battery_id="BAT-NB-001",
            category=BatteryCategory.EV,
            conformity_module=ConformityModule.MODULE_B,
            technical_documentation=all_docs_available,
            test_reports=test_reports,
            test_results=test_results,
            eu_declaration=all_declaration_available,
            ce_marking=True,
            notified_body_id="NB-1234",
            notified_body_certificate="CERT-001",
        )
        result = engine.assess_conformity(input_data)
        assert result.notified_body_required is True
        assert result.notified_body_engaged is True

    def test_quality_system_module_d(self, engine, all_docs_available, all_declaration_available, ev_tests_all_done):
        test_reports, test_results = ev_tests_all_done
        input_data = ConformityInput(
            battery_id="BAT-QS-001",
            category=BatteryCategory.EV,
            conformity_module=ConformityModule.MODULE_D,
            technical_documentation=all_docs_available,
            test_reports=test_reports,
            test_results=test_results,
            eu_declaration=all_declaration_available,
            ce_marking=True,
            notified_body_id="NB-1234",
            quality_system_certified=True,
            quality_system_standard="ISO 9001:2015",
        )
        result = engine.assess_conformity(input_data)
        assert result.quality_system_adequate is True

    def test_fully_ready_positive_recommendation(self, engine, fully_ready_ev_input):
        result = engine.assess_conformity(fully_ready_ev_input)
        assert any("ready" in r.lower() for r in result.recommendations)

    def test_no_missing_when_fully_ready(self, engine, fully_ready_ev_input):
        result = engine.assess_conformity(fully_ready_ev_input)
        assert len(result.missing_items) == 0


# ---------------------------------------------------------------------------
# Test: Module Info
# ---------------------------------------------------------------------------


class TestModuleInfo:
    def test_get_module_info_a(self, engine):
        info = engine.get_module_info(ConformityModule.MODULE_A)
        assert info["module"] == "module_a"
        assert info["requires_notified_body"] is False
        assert info["requires_quality_system"] is False
        assert "provenance_hash" in info

    def test_get_module_info_h(self, engine):
        info = engine.get_module_info(ConformityModule.MODULE_H)
        assert info["requires_notified_body"] is True
        assert info["requires_quality_system"] is True

    def test_get_module_info_b(self, engine):
        info = engine.get_module_info(ConformityModule.MODULE_B)
        assert info["requires_notified_body"] is True
        assert info["requires_quality_system"] is False

    def test_get_all_modules(self, engine):
        all_modules = engine.get_all_modules()
        assert len(all_modules) == len(ConformityModule)
        for module in ConformityModule:
            assert module.value in all_modules


# ---------------------------------------------------------------------------
# Test: Checklist Retrieval
# ---------------------------------------------------------------------------


class TestChecklistRetrieval:
    def test_get_documentation_checklist(self, engine):
        checklist = engine.get_documentation_checklist()
        assert len(checklist) == 12
        assert "td_01" in checklist

    def test_get_declaration_checklist(self, engine):
        checklist = engine.get_declaration_checklist()
        assert len(checklist) == 8
        assert "doc_01" in checklist

    def test_get_test_requirements_ev(self, engine):
        tests = engine.get_test_requirements(BatteryCategory.EV)
        assert len(tests) == 9

    def test_get_test_requirements_portable(self, engine):
        tests = engine.get_test_requirements(BatteryCategory.PORTABLE)
        assert len(tests) == 4

    def test_get_test_requirements_sli(self, engine):
        tests = engine.get_test_requirements(BatteryCategory.SLI)
        assert len(tests) == 4

    def test_get_test_requirements_lmt(self, engine):
        tests = engine.get_test_requirements(BatteryCategory.LMT)
        assert len(tests) == 5

    def test_get_test_requirements_industrial(self, engine):
        tests = engine.get_test_requirements(BatteryCategory.INDUSTRIAL)
        assert len(tests) == 7


# ---------------------------------------------------------------------------
# Test: Conformity Summary
# ---------------------------------------------------------------------------


class TestConformitySummary:
    def test_summary_structure_ready(self, engine, fully_ready_ev_input):
        result = engine.assess_conformity(fully_ready_ev_input)
        summary = engine.get_conformity_summary(result)
        assert summary["battery_id"] == "BAT-READY-001"
        assert summary["category"] == "ev"
        assert summary["module"] == "module_a"
        assert summary["overall_status"] == "ready"
        assert summary["overall_score"] == 100.0
        assert summary["missing_items_count"] == 0
        assert "provenance_hash" in summary

    def test_summary_structure_not_ready(self, engine, empty_ev_input):
        result = engine.assess_conformity(empty_ev_input)
        summary = engine.get_conformity_summary(result)
        assert summary["overall_status"] == "not_ready"
        assert summary["missing_items_count"] > 0


# ---------------------------------------------------------------------------
# Test: Missing Items Identification
# ---------------------------------------------------------------------------


class TestMissingItems:
    def test_ce_marking_missing(self, engine, empty_ev_input):
        result = engine.assess_conformity(empty_ev_input)
        assert any("CE marking" in m for m in result.missing_items)

    def test_documentation_missing(self, engine, empty_ev_input):
        result = engine.assess_conformity(empty_ev_input)
        assert any("Technical documentation" in m for m in result.missing_items)

    def test_tests_missing(self, engine, empty_ev_input):
        result = engine.assess_conformity(empty_ev_input)
        assert any("Test not completed" in m for m in result.missing_items)

    def test_declaration_missing(self, engine, empty_ev_input):
        result = engine.assess_conformity(empty_ev_input)
        assert any("EU Declaration" in m for m in result.missing_items)

    def test_no_missing_for_fully_ready(self, engine, fully_ready_ev_input):
        result = engine.assess_conformity(fully_ready_ev_input)
        assert len(result.missing_items) == 0


# ---------------------------------------------------------------------------
# Test: Recommendations
# ---------------------------------------------------------------------------


class TestRecommendations:
    def test_ready_positive_message(self, engine, fully_ready_ev_input):
        result = engine.assess_conformity(fully_ready_ev_input)
        assert len(result.recommendations) == 1
        assert "ready" in result.recommendations[0].lower()

    def test_not_ready_has_doc_recommendation(self, engine, empty_ev_input):
        result = engine.assess_conformity(empty_ev_input)
        assert any("Technical documentation" in r for r in result.recommendations)

    def test_critical_doc_below_50(self, engine, empty_ev_input):
        result = engine.assess_conformity(empty_ev_input)
        assert any("CRITICAL" in r for r in result.recommendations)

    def test_ce_marking_recommendation(self, engine, empty_ev_input):
        result = engine.assess_conformity(empty_ev_input)
        assert any("CE marking" in r for r in result.recommendations)

    def test_test_recommendation(self, engine, empty_ev_input):
        result = engine.assess_conformity(empty_ev_input)
        assert any("test" in r.lower() for r in result.recommendations)

    def test_ev_carbon_footprint_recommendation(self, engine):
        # EV with all docs except carbon footprint
        docs = {item_id: True for item_id in TECHNICAL_DOC_CHECKLIST}
        docs["td_08"] = False
        input_data = ConformityInput(
            battery_id="BAT-CF-001",
            category=BatteryCategory.EV,
            technical_documentation=docs,
        )
        result = engine.assess_conformity(input_data)
        assert any("Carbon footprint" in r for r in result.recommendations)

    def test_nb_recommendation_when_required(self, engine):
        input_data = ConformityInput(
            battery_id="BAT-NB-001",
            category=BatteryCategory.EV,
            conformity_module=ConformityModule.MODULE_B,
        )
        result = engine.assess_conformity(input_data)
        assert any("notified body" in r.lower() for r in result.recommendations)

    def test_failed_test_recommendation(self, engine):
        ev_tests = TEST_REQUIREMENTS["ev"]
        test_reports = {ev_tests[0]["test"]: True}
        test_results = {ev_tests[0]["test"]: False}
        input_data = ConformityInput(
            battery_id="BAT-FAIL-001",
            category=BatteryCategory.EV,
            test_reports=test_reports,
            test_results=test_results,
        )
        result = engine.assess_conformity(input_data)
        assert any("failed" in r.lower() for r in result.recommendations)


# ---------------------------------------------------------------------------
# Test: Provenance / Determinism
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_hash_exists(self, engine, fully_ready_ev_input):
        result = engine.assess_conformity(fully_ready_ev_input)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_module_info_has_hash(self, engine):
        info = engine.get_module_info(ConformityModule.MODULE_A)
        assert len(info["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# Test: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_categories_assessable(self, engine):
        for cat in BatteryCategory:
            input_data = ConformityInput(
                battery_id=f"BAT-{cat.value}",
                category=cat,
            )
            result = engine.assess_conformity(input_data)
            assert isinstance(result, ConformityResult)

    def test_all_modules_assessable(self, engine):
        for module in ConformityModule:
            input_data = ConformityInput(
                battery_id=f"BAT-{module.value}",
                category=BatteryCategory.EV,
                conformity_module=module,
            )
            result = engine.assess_conformity(input_data)
            assert result.module == module

    def test_extra_docs_not_in_checklist_ignored(self, engine):
        docs = {item_id: True for item_id in TECHNICAL_DOC_CHECKLIST}
        docs["extra_item"] = True  # Should be ignored
        items, completeness = engine.check_documentation(
            docs, BatteryCategory.EV
        )
        assert completeness == 100.0
        assert len(items) == 12

    def test_extra_tests_not_in_requirements_ignored(self, engine):
        test_reports = {"nonexistent test": True}
        test_results = {"nonexistent test": True}
        items, coverage, pass_rate = engine.check_test_coverage(
            BatteryCategory.EV, test_reports, test_results
        )
        assert coverage == 0.0  # None of the required tests done
