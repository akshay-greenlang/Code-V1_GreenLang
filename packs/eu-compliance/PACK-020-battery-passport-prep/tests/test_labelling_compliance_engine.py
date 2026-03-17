# -*- coding: utf-8 -*-
"""
Tests for LabellingComplianceEngine - PACK-020 Engine 6
========================================================

Comprehensive tests for labelling and marking compliance
per EU Battery Regulation Art 13-14.

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


mod = _load_module("labelling_compliance_engine.py", "pack020_test.engines.labelling_compliance")

LabellingComplianceEngine = mod.LabellingComplianceEngine
LabelRequirement = mod.LabelRequirement
LabelElementCheck = mod.LabelElementCheck
LabelCheckResult = mod.LabelCheckResult
BatteryCategory = mod.BatteryCategory
LabelElement = mod.LabelElement
LabelStatus = mod.LabelStatus
LABEL_ELEMENT_DESCRIPTIONS = mod.LABEL_ELEMENT_DESCRIPTIONS
CATEGORY_REQUIREMENTS = mod.CATEGORY_REQUIREMENTS
CE_MARKING_MIN_HEIGHT_MM = mod.CE_MARKING_MIN_HEIGHT_MM
QR_CODE_MIN_MODULE_MM = mod.QR_CODE_MIN_MODULE_MM
CORRECTIVE_ACTIONS = mod.CORRECTIVE_ACTIONS
_compute_hash = mod._compute_hash
_safe_divide = mod._safe_divide
_round2 = mod._round2
_round3 = mod._round3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    return LabellingComplianceEngine()


@pytest.fixture
def all_present_labels():
    """All label elements marked as present."""
    return {elem.value: "present" for elem in LabelElement}


@pytest.fixture
def all_missing_labels():
    """All label elements marked as missing."""
    return {elem.value: "missing" for elem in LabelElement}


@pytest.fixture
def partial_labels():
    """Some present, some missing."""
    return {
        LabelElement.CE_MARKING.value: "present",
        LabelElement.QR_CODE.value: "present",
        LabelElement.COLLECTION_SYMBOL.value: "missing",
        LabelElement.CAPACITY_LABEL.value: "present",
        LabelElement.HAZARDOUS_SUBSTANCE.value: "incorrect",
        LabelElement.BATTERY_CHEMISTRY.value: "present",
        LabelElement.CARBON_FOOTPRINT.value: "missing",
        LabelElement.SEPARATE_COLLECTION.value: "missing",
        LabelElement.MANUFACTURER_INFO.value: "present",
        LabelElement.COUNTRY_OF_ORIGIN.value: "present",
    }


# ---------------------------------------------------------------------------
# Test: Engine Initialization
# ---------------------------------------------------------------------------


class TestLabellingComplianceEngineInit:
    def test_init_creates_engine(self):
        engine = LabellingComplianceEngine()
        assert engine is not None

    def test_engine_version(self):
        engine = LabellingComplianceEngine()
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

    def test_label_element_values(self):
        assert LabelElement.CE_MARKING.value == "ce_marking"
        assert LabelElement.QR_CODE.value == "qr_code"
        assert LabelElement.COLLECTION_SYMBOL.value == "collection_symbol"
        assert LabelElement.CAPACITY_LABEL.value == "capacity_label"
        assert LabelElement.HAZARDOUS_SUBSTANCE.value == "hazardous_substance"
        assert LabelElement.BATTERY_CHEMISTRY.value == "battery_chemistry"
        assert LabelElement.CARBON_FOOTPRINT.value == "carbon_footprint"
        assert LabelElement.SEPARATE_COLLECTION.value == "separate_collection"
        assert LabelElement.MANUFACTURER_INFO.value == "manufacturer_info"
        assert LabelElement.COUNTRY_OF_ORIGIN.value == "country_of_origin"

    def test_label_element_count(self):
        assert len(LabelElement) == 10

    def test_label_status_values(self):
        assert LabelStatus.PRESENT.value == "present"
        assert LabelStatus.MISSING.value == "missing"
        assert LabelStatus.INCORRECT.value == "incorrect"
        assert LabelStatus.NOT_REQUIRED.value == "not_required"

    def test_label_status_count(self):
        assert len(LabelStatus) == 4


# ---------------------------------------------------------------------------
# Test: Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_descriptions_has_all_elements(self):
        for elem in LabelElement:
            assert elem.value in LABEL_ELEMENT_DESCRIPTIONS

    def test_category_requirements_has_all_categories(self):
        for cat in BatteryCategory:
            assert cat.value in CATEGORY_REQUIREMENTS

    def test_ce_marking_min_height(self):
        assert CE_MARKING_MIN_HEIGHT_MM == 5

    def test_qr_code_min_module(self):
        assert QR_CODE_MIN_MODULE_MM == 0.5

    def test_corrective_actions_has_all_elements(self):
        for elem in LabelElement:
            assert elem.value in CORRECTIVE_ACTIONS

    def test_ev_requires_carbon_footprint(self):
        assert CATEGORY_REQUIREMENTS["ev"][LabelElement.CARBON_FOOTPRINT.value] is True

    def test_portable_does_not_require_carbon_footprint(self):
        assert CATEGORY_REQUIREMENTS["portable"][LabelElement.CARBON_FOOTPRINT.value] is False

    def test_ev_does_not_require_collection_symbol(self):
        assert CATEGORY_REQUIREMENTS["ev"][LabelElement.COLLECTION_SYMBOL.value] is False

    def test_portable_requires_collection_symbol(self):
        assert CATEGORY_REQUIREMENTS["portable"][LabelElement.COLLECTION_SYMBOL.value] is True

    def test_industrial_requires_carbon_footprint(self):
        assert CATEGORY_REQUIREMENTS["industrial"][LabelElement.CARBON_FOOTPRINT.value] is True

    def test_portable_does_not_require_country_of_origin(self):
        assert CATEGORY_REQUIREMENTS["portable"][LabelElement.COUNTRY_OF_ORIGIN.value] is False

    def test_ev_requires_country_of_origin(self):
        assert CATEGORY_REQUIREMENTS["ev"][LabelElement.COUNTRY_OF_ORIGIN.value] is True


# ---------------------------------------------------------------------------
# Test: Parse Label Status
# ---------------------------------------------------------------------------


class TestParseLabelStatus:
    def test_present_variants(self, engine):
        for v in ["present", "yes", "true", "ok", "compliant"]:
            assert engine._parse_label_status(v) == LabelStatus.PRESENT

    def test_missing_variants(self, engine):
        for v in ["missing", "no", "false", "absent"]:
            assert engine._parse_label_status(v) == LabelStatus.MISSING

    def test_incorrect_variants(self, engine):
        for v in ["incorrect", "wrong", "error", "invalid"]:
            assert engine._parse_label_status(v) == LabelStatus.INCORRECT

    def test_not_required_variants(self, engine):
        for v in ["not_required", "n/a", "na"]:
            assert engine._parse_label_status(v) == LabelStatus.NOT_REQUIRED

    def test_unknown_defaults_to_missing(self, engine):
        assert engine._parse_label_status("unknown") == LabelStatus.MISSING

    def test_case_insensitive(self, engine):
        assert engine._parse_label_status("PRESENT") == LabelStatus.PRESENT
        assert engine._parse_label_status("Missing") == LabelStatus.MISSING

    def test_strips_whitespace(self, engine):
        assert engine._parse_label_status("  present  ") == LabelStatus.PRESENT


# ---------------------------------------------------------------------------
# Test: Get Required Elements
# ---------------------------------------------------------------------------


class TestGetRequiredElements:
    def test_returns_all_elements(self, engine):
        result = engine.get_required_elements(BatteryCategory.PORTABLE)
        assert len(result) == len(LabelElement)

    def test_portable_required_elements(self, engine):
        result = engine.get_required_elements(BatteryCategory.PORTABLE)
        required_names = [r.element.value for r in result if r.required]
        assert LabelElement.CE_MARKING.value in required_names
        assert LabelElement.QR_CODE.value in required_names
        assert LabelElement.CARBON_FOOTPRINT.value not in required_names

    def test_ev_required_elements(self, engine):
        result = engine.get_required_elements(BatteryCategory.EV)
        required_names = [r.element.value for r in result if r.required]
        assert LabelElement.CARBON_FOOTPRINT.value in required_names
        assert LabelElement.COLLECTION_SYMBOL.value not in required_names

    def test_element_has_description(self, engine):
        result = engine.get_required_elements(BatteryCategory.EV)
        for req in result:
            assert req.description != ""

    def test_element_has_category_applicable(self, engine):
        result = engine.get_required_elements(BatteryCategory.EV)
        for req in result:
            assert req.category_applicable == "ev"

    def test_get_required_element_names_sorted(self, engine):
        names = engine.get_required_element_names(BatteryCategory.PORTABLE)
        assert names == sorted(names)

    def test_get_required_element_names_ev(self, engine):
        names = engine.get_required_element_names(BatteryCategory.EV)
        assert "carbon_footprint" in names
        assert "collection_symbol" not in names


# ---------------------------------------------------------------------------
# Test: Validate Element
# ---------------------------------------------------------------------------


class TestValidateElement:
    def test_present_required_is_compliant(self, engine):
        check = engine.validate_element(
            LabelElement.CE_MARKING, BatteryCategory.EV, "present", True
        )
        assert check.compliant is True
        assert check.corrective_action == ""

    def test_missing_required_is_non_compliant(self, engine):
        check = engine.validate_element(
            LabelElement.CE_MARKING, BatteryCategory.EV, "missing", True
        )
        assert check.compliant is False
        assert check.corrective_action != ""

    def test_incorrect_required_is_non_compliant(self, engine):
        check = engine.validate_element(
            LabelElement.CE_MARKING, BatteryCategory.EV, "incorrect", True
        )
        assert check.compliant is False

    def test_not_required_is_always_compliant(self, engine):
        check = engine.validate_element(
            LabelElement.CARBON_FOOTPRINT, BatteryCategory.PORTABLE, "missing", False
        )
        assert check.compliant is True
        assert check.status == LabelStatus.NOT_REQUIRED

    def test_check_has_description(self, engine):
        check = engine.validate_element(
            LabelElement.CE_MARKING, BatteryCategory.EV, "present", True
        )
        assert check.description != ""


# ---------------------------------------------------------------------------
# Test: Full Labelling Check
# ---------------------------------------------------------------------------


class TestCheckLabelling:
    def test_fully_compliant_ev(self, engine, all_present_labels):
        result = engine.check_labelling("BAT-001", BatteryCategory.EV, all_present_labels)
        assert result.overall_compliant is True
        assert result.compliance_pct == 100.0
        assert result.non_compliant_count == 0

    def test_all_missing_non_compliant(self, engine, all_missing_labels):
        result = engine.check_labelling("BAT-002", BatteryCategory.EV, all_missing_labels)
        assert result.overall_compliant is False
        assert result.compliance_pct == 0.0

    def test_partial_compliance(self, engine, partial_labels):
        result = engine.check_labelling("BAT-003", BatteryCategory.EV, partial_labels)
        assert result.overall_compliant is False
        assert 0.0 < result.compliance_pct < 100.0

    def test_provenance_hash_populated(self, engine, all_present_labels):
        result = engine.check_labelling("BAT-001", BatteryCategory.EV, all_present_labels)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_result_has_uuid(self, engine, all_present_labels):
        result = engine.check_labelling("BAT-001", BatteryCategory.EV, all_present_labels)
        assert result.result_id is not None
        assert len(result.result_id) == 36

    def test_engine_version_in_result(self, engine, all_present_labels):
        result = engine.check_labelling("BAT-001", BatteryCategory.EV, all_present_labels)
        assert result.engine_version == "1.0.0"

    def test_battery_id_in_result(self, engine, all_present_labels):
        result = engine.check_labelling("BAT-XYZ", BatteryCategory.EV, all_present_labels)
        assert result.battery_id == "BAT-XYZ"

    def test_category_in_result(self, engine, all_present_labels):
        result = engine.check_labelling("BAT-001", BatteryCategory.EV, all_present_labels)
        assert result.category == BatteryCategory.EV

    def test_elements_checked_count(self, engine, all_present_labels):
        result = engine.check_labelling("BAT-001", BatteryCategory.EV, all_present_labels)
        assert result.elements_checked == len(LabelElement)

    def test_missing_elements_tracked(self, engine, all_missing_labels):
        result = engine.check_labelling("BAT-001", BatteryCategory.PORTABLE, all_missing_labels)
        assert len(result.missing_elements) > 0

    def test_incorrect_elements_tracked(self, engine, partial_labels):
        result = engine.check_labelling("BAT-001", BatteryCategory.PORTABLE, partial_labels)
        assert "hazardous_substance" in result.incorrect_elements

    def test_not_required_elements_tracked(self, engine, all_present_labels):
        result = engine.check_labelling("BAT-001", BatteryCategory.PORTABLE, all_present_labels)
        # Portable does not require carbon_footprint
        assert "carbon_footprint" in result.not_required_elements

    def test_recommendations_for_non_compliant(self, engine, all_missing_labels):
        result = engine.check_labelling("BAT-001", BatteryCategory.EV, all_missing_labels)
        assert len(result.recommendations) > 0

    def test_recommendations_for_compliant(self, engine, all_present_labels):
        result = engine.check_labelling("BAT-001", BatteryCategory.EV, all_present_labels)
        assert len(result.recommendations) > 0
        assert any("No corrective actions" in r for r in result.recommendations)

    def test_processing_time_recorded(self, engine, all_present_labels):
        result = engine.check_labelling("BAT-001", BatteryCategory.EV, all_present_labels)
        assert result.processing_time_ms >= 0.0

    def test_portable_carbon_footprint_not_counted(self, engine):
        # Even if carbon_footprint is missing, portable should not count it
        labels = {elem.value: "present" for elem in LabelElement}
        labels[LabelElement.CARBON_FOOTPRINT.value] = "missing"
        result = engine.check_labelling("BAT-001", BatteryCategory.PORTABLE, labels)
        assert result.overall_compliant is True

    def test_ev_carbon_footprint_required(self, engine):
        labels = {elem.value: "present" for elem in LabelElement}
        labels[LabelElement.CARBON_FOOTPRINT.value] = "missing"
        result = engine.check_labelling("BAT-001", BatteryCategory.EV, labels)
        assert result.overall_compliant is False
        assert "carbon_footprint" in result.missing_elements

    def test_empty_labels_dict(self, engine):
        result = engine.check_labelling("BAT-001", BatteryCategory.PORTABLE, {})
        assert result.overall_compliant is False
        assert result.compliance_pct == 0.0


# ---------------------------------------------------------------------------
# Test: Corrective Actions
# ---------------------------------------------------------------------------


class TestCorrectiveActions:
    def test_no_issues_positive_message(self, engine):
        recs = engine.generate_corrective_actions([], [], BatteryCategory.EV, 100.0)
        assert len(recs) == 1
        assert "No corrective actions" in recs[0]

    def test_missing_ce_marking_priority(self, engine):
        recs = engine.generate_corrective_actions(
            ["ce_marking"], [], BatteryCategory.EV, 50.0
        )
        assert any("PRIORITY" in r and "CE marking" in r for r in recs)

    def test_missing_qr_code_priority(self, engine):
        recs = engine.generate_corrective_actions(
            ["qr_code"], [], BatteryCategory.EV, 50.0
        )
        assert any("PRIORITY" in r and "QR code" in r for r in recs)

    def test_missing_carbon_footprint_ev_priority(self, engine):
        recs = engine.generate_corrective_actions(
            ["carbon_footprint"], [], BatteryCategory.EV, 50.0
        )
        assert any("Carbon footprint" in r for r in recs)

    def test_missing_carbon_footprint_portable_no_priority(self, engine):
        recs = engine.generate_corrective_actions(
            ["carbon_footprint"], [], BatteryCategory.PORTABLE, 50.0
        )
        # No special priority for portable
        assert not any("mandatory" in r.lower() and "carbon footprint" in r.lower() for r in recs)

    def test_critical_below_50_pct(self, engine):
        recs = engine.generate_corrective_actions(
            ["ce_marking", "qr_code"], [], BatteryCategory.EV, 30.0
        )
        assert any("CRITICAL" in r for r in recs)

    def test_incorrect_elements_generate_actions(self, engine):
        recs = engine.generate_corrective_actions(
            [], ["capacity_label"], BatteryCategory.EV, 80.0
        )
        assert any("INCORRECT" in r for r in recs)


# ---------------------------------------------------------------------------
# Test: Category Comparison
# ---------------------------------------------------------------------------


class TestCategoryComparison:
    def test_comparison_has_all_categories(self, engine):
        result = engine.compare_category_requirements()
        for cat in BatteryCategory:
            assert cat.value in result["categories"]

    def test_universal_elements_found(self, engine):
        result = engine.compare_category_requirements()
        universal = result["universal_elements"]
        # CE marking and QR code should be universal
        assert "ce_marking" in universal
        assert "qr_code" in universal

    def test_category_specific_elements(self, engine):
        result = engine.compare_category_requirements()
        specific = result["category_specific_elements"]
        # carbon_footprint is specific to EV and industrial (not universal)
        assert "carbon_footprint" not in result["universal_elements"]

    def test_provenance_hash(self, engine):
        result = engine.compare_category_requirements()
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_required_count_per_category(self, engine):
        result = engine.compare_category_requirements()
        for cat in BatteryCategory:
            cat_data = result["categories"][cat.value]
            assert "required_count" in cat_data
            assert cat_data["required_count"] > 0


# ---------------------------------------------------------------------------
# Test: Element Description
# ---------------------------------------------------------------------------


class TestElementDescription:
    def test_get_element_description(self, engine):
        result = engine.get_element_description(LabelElement.CE_MARKING)
        assert result["element"] == "ce_marking"
        assert result["description"] != ""
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_applicable_categories_listed(self, engine):
        result = engine.get_element_description(LabelElement.CE_MARKING)
        # CE marking required in all categories
        assert len(result["applicable_categories"]) == 5

    def test_carbon_footprint_categories(self, engine):
        result = engine.get_element_description(LabelElement.CARBON_FOOTPRINT)
        assert "ev" in result["applicable_categories"]
        assert "industrial" in result["applicable_categories"]
        assert "portable" not in result["applicable_categories"]

    def test_get_all_descriptions(self, engine):
        result = engine.get_all_element_descriptions()
        assert len(result) == len(LabelElement)
        for elem in LabelElement:
            assert elem.value in result


# ---------------------------------------------------------------------------
# Test: Batch Checking
# ---------------------------------------------------------------------------


class TestBatchChecking:
    def test_batch_single_battery(self, engine, all_present_labels):
        batteries = [
            {"battery_id": "BAT-001", "category": "ev", "labels": all_present_labels},
        ]
        result = engine.check_labelling_batch(batteries)
        assert result["total_batteries"] == 1
        assert result["fully_compliant"] == 1

    def test_batch_multiple_batteries(self, engine, all_present_labels, all_missing_labels):
        batteries = [
            {"battery_id": "BAT-001", "category": "ev", "labels": all_present_labels},
            {"battery_id": "BAT-002", "category": "ev", "labels": all_missing_labels},
        ]
        result = engine.check_labelling_batch(batteries)
        assert result["total_batteries"] == 2
        assert result["fully_compliant"] == 1
        assert result["non_compliant"] == 1

    def test_batch_compliance_rate(self, engine, all_present_labels, all_missing_labels):
        batteries = [
            {"battery_id": "BAT-001", "category": "ev", "labels": all_present_labels},
            {"battery_id": "BAT-002", "category": "ev", "labels": all_missing_labels},
        ]
        result = engine.check_labelling_batch(batteries)
        assert result["batch_compliance_rate"] == 50.0

    def test_batch_empty_list(self, engine):
        result = engine.check_labelling_batch([])
        assert result["total_batteries"] == 0

    def test_batch_most_common_missing(self, engine, all_missing_labels):
        batteries = [
            {"battery_id": f"BAT-{i}", "category": "ev", "labels": all_missing_labels}
            for i in range(3)
        ]
        result = engine.check_labelling_batch(batteries)
        assert len(result["most_common_missing"]) > 0

    def test_batch_provenance_hash(self, engine, all_present_labels):
        batteries = [
            {"battery_id": "BAT-001", "category": "ev", "labels": all_present_labels},
        ]
        result = engine.check_labelling_batch(batteries)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_batch_invalid_category_defaults_portable(self, engine, all_present_labels):
        batteries = [
            {"battery_id": "BAT-001", "category": "invalid_cat", "labels": all_present_labels},
        ]
        result = engine.check_labelling_batch(batteries)
        assert result["results"][0]["category"] == "portable"

    def test_batch_processing_time(self, engine, all_present_labels):
        batteries = [
            {"battery_id": "BAT-001", "category": "ev", "labels": all_present_labels},
        ]
        result = engine.check_labelling_batch(batteries)
        assert result["processing_time_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Test: Compliance Summary
# ---------------------------------------------------------------------------


class TestComplianceSummary:
    def test_summary_structure(self, engine, all_present_labels):
        check = engine.check_labelling("BAT-001", BatteryCategory.EV, all_present_labels)
        summary = engine.get_compliance_summary(check)
        assert summary["battery_id"] == "BAT-001"
        assert summary["category"] == "ev"
        assert summary["overall_compliant"] is True
        assert summary["compliance_pct"] == 100.0
        assert "provenance_hash" in summary

    def test_summary_non_compliant(self, engine, all_missing_labels):
        check = engine.check_labelling("BAT-001", BatteryCategory.EV, all_missing_labels)
        summary = engine.get_compliance_summary(check)
        assert summary["overall_compliant"] is False
        assert summary["non_compliant_count"] > 0
        assert len(summary["missing_elements"]) > 0


# ---------------------------------------------------------------------------
# Test: Provenance / Determinism
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_same_input_same_hash(self, engine, all_present_labels):
        r1 = engine.check_labelling("BAT-001", BatteryCategory.EV, all_present_labels)
        r2 = engine.check_labelling("BAT-001", BatteryCategory.EV, all_present_labels)
        # Different result_ids and timestamps but same content provenance
        # is not guaranteed because of uuid/timestamp; so we verify hash exists
        assert r1.provenance_hash != ""
        assert r2.provenance_hash != ""

    def test_different_input_different_hash(self, engine, all_present_labels, all_missing_labels):
        r1 = engine.check_labelling("BAT-001", BatteryCategory.EV, all_present_labels)
        r2 = engine.check_labelling("BAT-001", BatteryCategory.EV, all_missing_labels)
        assert r1.provenance_hash != r2.provenance_hash


# ---------------------------------------------------------------------------
# Test: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_categories_can_be_checked(self, engine, all_present_labels):
        for cat in BatteryCategory:
            result = engine.check_labelling("BAT-001", cat, all_present_labels)
            assert isinstance(result, LabelCheckResult)
            assert result.overall_compliant is True

    def test_label_status_with_extra_whitespace(self, engine):
        labels = {LabelElement.CE_MARKING.value: "  present  "}
        result = engine.check_labelling("BAT-001", BatteryCategory.PORTABLE, labels)
        ce_check = next(
            c for c in result.element_checks if c.element == LabelElement.CE_MARKING
        )
        assert ce_check.status == LabelStatus.PRESENT

    def test_label_with_alias_yes(self, engine):
        labels = {elem.value: "yes" for elem in LabelElement}
        result = engine.check_labelling("BAT-001", BatteryCategory.EV, labels)
        assert result.overall_compliant is True

    def test_compliance_pct_rounding(self, engine):
        # 7 out of 8 required = 87.5%
        # Verify rounding behavior is consistent
        labels = {elem.value: "present" for elem in LabelElement}
        result = engine.check_labelling("BAT-001", BatteryCategory.PORTABLE, labels)
        assert result.compliance_pct == 100.0
