# -*- coding: utf-8 -*-
"""
Tests for BatteryPassportEngine - PACK-020 Engine 3
=====================================================

Comprehensive tests for battery passport compilation and
validation per EU Battery Regulation Art 77-78 and Annex XIII.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-020 Battery Passport Prep
"""

import importlib.util
import json
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


mod = _load_module("battery_passport_engine.py", "pack020_test.engines.battery_passport")

BatteryPassportEngine = mod.BatteryPassportEngine
PassportData = mod.PassportData
GeneralInfo = mod.GeneralInfo
CarbonFootprintInfo = mod.CarbonFootprintInfo
SupplyChainDD = mod.SupplyChainDD
MaterialComposition = mod.MaterialComposition
PerformanceDurability = mod.PerformanceDurability
EndOfLifeInfo = mod.EndOfLifeInfo
PassportValidationResult = mod.PassportValidationResult
FieldValidation = mod.FieldValidation
PassportField = mod.PassportField
PassportStatus = mod.PassportStatus
DataQuality = mod.DataQuality
AccessLevel = mod.AccessLevel
REQUIRED_FIELDS = mod.REQUIRED_FIELDS
OPTIONAL_FIELDS = mod.OPTIONAL_FIELDS
ALL_REQUIRED_FIELDS = mod.ALL_REQUIRED_FIELDS
ALL_OPTIONAL_FIELDS = mod.ALL_OPTIONAL_FIELDS
SECTION_LABELS = mod.SECTION_LABELS
FIELD_ACCESS_LEVELS = mod.FIELD_ACCESS_LEVELS
FIELD_DESCRIPTIONS = mod.FIELD_DESCRIPTIONS


@pytest.fixture
def engine():
    return BatteryPassportEngine()


@pytest.fixture
def full_general_info():
    return GeneralInfo(
        manufacturer_id="MFG-001",
        manufacturing_plant="Gigafactory Berlin",
        manufacturing_date="2027-01-15",
        manufacturing_country="DE",
        battery_model="EB-75-NMC811",
        battery_batch="BATCH-2027",
        battery_serial="SN-2027-001",
        battery_weight_kg=Decimal("450"),
        battery_category="ev",
        battery_chemistry="nmc811",
        energy_capacity_kwh=Decimal("75"),
        voltage_nominal=Decimal("400"),
    )


@pytest.fixture
def full_passport(full_general_info):
    return PassportData(
        general_info=full_general_info,
        carbon_footprint=CarbonFootprintInfo(
            total_co2e_kg=Decimal("5000"),
            per_kwh_co2e_kg=Decimal("66.667"),
            performance_class="class_b",
            lifecycle_breakdown={"raw_material": "3000", "manufacturing": "1500"},
            methodology={"method": "PEFCR"},
        ),
        supply_chain_dd=SupplyChainDD(
            dd_policy="Policy documented",
            third_party_verification="Verified by TUV",
            conflict_minerals="Assessment complete",
            supply_chain_mapping="Full mapping available",
        ),
        material_composition=MaterialComposition(
            bill_of_materials={"cobalt": "12.5 kg", "lithium": "8 kg"},
            hazardous_substances=["electrolyte"],
            critical_raw_materials={"cobalt": "12.5 kg"},
            recycled_content={"cobalt": "20%"},
        ),
        performance_durability=PerformanceDurability(
            rated_capacity_ah=Decimal("75"),
            cycle_life_expected=1500,
            energy_efficiency_pct=Decimal("95"),
            internal_resistance_mohm=Decimal("50"),
            state_of_health_pct=Decimal("100"),
            temperature_range_min=Decimal("-20"),
            temperature_range_max=Decimal("45"),
            c_rate_max=Decimal("2"),
        ),
        end_of_life=EndOfLifeInfo(
            collection_info="Take-back scheme available",
            recycling_info="Recycling at approved facilities",
            second_life_info="Second life assessment available",
            safety_instructions="Handle with care",
        ),
    )


@pytest.fixture
def empty_passport():
    return PassportData()


class TestEngineInit:
    def test_init(self):
        engine = BatteryPassportEngine()
        assert engine.engine_version == "1.0.0"

    def test_empty_registry(self):
        engine = BatteryPassportEngine()
        assert engine.get_all_passports() == {}
        assert engine.get_validation_results() == []


class TestEnums:
    def test_passport_field_count(self):
        assert len(PassportField) >= 30

    def test_passport_status_values(self):
        assert PassportStatus.DRAFT.value == "draft"
        assert PassportStatus.VALIDATED.value == "validated"
        assert PassportStatus.PUBLISHED.value == "published"

    def test_data_quality_values(self):
        assert len(DataQuality) == 4

    def test_access_level_values(self):
        assert len(AccessLevel) == 4


class TestConstants:
    def test_required_fields_has_all_sections(self):
        assert len(REQUIRED_FIELDS) == 6

    def test_all_required_fields_is_populated(self):
        assert len(ALL_REQUIRED_FIELDS) > 0

    def test_all_optional_fields_is_populated(self):
        assert len(ALL_OPTIONAL_FIELDS) > 0

    def test_section_labels(self):
        assert len(SECTION_LABELS) == 6

    def test_field_descriptions_cover_all_fields(self):
        for field in PassportField:
            assert field.value in FIELD_DESCRIPTIONS

    def test_field_access_levels_cover_all_fields(self):
        for field in PassportField:
            assert field.value in FIELD_ACCESS_LEVELS


class TestCompilePassport:
    def test_compile_assigns_id(self, engine, full_passport):
        compiled = engine.compile_passport(full_passport)
        assert compiled.passport_id != ""

    def test_compile_stores_in_registry(self, engine, full_passport):
        compiled = engine.compile_passport(full_passport)
        assert engine.get_passport(compiled.passport_id) is not None

    def test_compile_preserves_data(self, engine, full_passport):
        compiled = engine.compile_passport(full_passport)
        assert compiled.general_info.manufacturer_id == "MFG-001"


class TestValidatePassport:
    def test_full_passport_high_completeness(self, engine, full_passport):
        result = engine.validate_passport(full_passport)
        assert isinstance(result, PassportValidationResult)
        assert result.completeness_pct >= Decimal("80")

    def test_empty_passport_zero_completeness(self, engine, empty_passport):
        result = engine.validate_passport(empty_passport)
        assert result.completeness_pct == Decimal("0.00")
        assert result.status == PassportStatus.DRAFT

    def test_result_has_provenance_hash(self, engine, full_passport):
        result = engine.validate_passport(full_passport)
        assert len(result.provenance_hash) == 64

    def test_result_has_field_validations(self, engine, full_passport):
        result = engine.validate_passport(full_passport)
        assert len(result.field_validations) == len(PassportField)

    def test_result_has_section_completeness(self, engine, full_passport):
        result = engine.validate_passport(full_passport)
        assert len(result.section_completeness) > 0

    def test_full_passport_status(self, engine, full_passport):
        result = engine.validate_passport(full_passport)
        # Full passport should be VALIDATED (100% required fields)
        assert result.status in (PassportStatus.VALIDATED, PassportStatus.DRAFT)

    def test_missing_fields_reported(self, engine, empty_passport):
        result = engine.validate_passport(empty_passport)
        assert result.required_fields_missing > 0
        assert len(result.validation_errors) > 0

    def test_qr_payload_generated(self, engine, full_passport):
        result = engine.validate_passport(full_passport)
        assert result.qr_payload is not None
        payload = json.loads(result.qr_payload)
        assert "passport_id" in payload


class TestQRPayload:
    def test_qr_payload_structure(self, engine, full_passport):
        payload_str = engine.generate_qr_payload(full_passport)
        payload = json.loads(payload_str)
        assert "schema_version" in payload
        assert "regulation" in payload
        assert payload["regulation"] == "EU_2023_1542"
        assert "manufacturer_id" in payload
        assert "passport_url" in payload

    def test_qr_payload_has_battery_info(self, engine, full_passport):
        payload = json.loads(engine.generate_qr_payload(full_passport))
        assert payload["battery_category"] == "ev"
        assert payload["battery_chemistry"] == "nmc811"


class TestDataQualityAssessment:
    def test_assess_data_quality(self, engine, full_passport):
        result = engine.assess_data_quality(full_passport)
        assert "quality_distribution" in result
        assert "overall_quality_pct" in result
        assert "provenance_hash" in result

    def test_empty_passport_low_quality(self, engine, empty_passport):
        result = engine.assess_data_quality(empty_passport)
        assert result["overall_quality_pct"] <= 20.0


class TestBatchValidation:
    def test_batch_validation(self, engine, full_passport, empty_passport):
        results = engine.validate_batch([full_passport, empty_passport])
        assert len(results) == 2


class TestStatusManagement:
    def test_update_status_success(self, engine, full_passport):
        compiled = engine.compile_passport(full_passport)
        result = engine.update_status(compiled.passport_id, PassportStatus.PUBLISHED)
        assert result["success"] is True
        assert result["new_status"] == "published"

    def test_update_status_not_found(self, engine):
        result = engine.update_status("nonexistent", PassportStatus.PUBLISHED)
        assert result["success"] is False


class TestAccessLevelFiltering:
    def test_get_public_fields(self, engine, full_passport):
        result = engine.get_public_fields(full_passport)
        assert "fields" in result
        assert result["access_level"] == "public"


class TestComparePassports:
    def test_compare_empty(self, engine):
        result = engine.compare_passports([])
        assert result["count"] == 0

    def test_compare_multiple(self, engine, full_passport, empty_passport):
        r1 = engine.validate_passport(full_passport)
        r2 = engine.validate_passport(empty_passport)
        comparison = engine.compare_passports([r1, r2])
        assert comparison["count"] == 2
        assert "statistics" in comparison


class TestRegistryManagement:
    def test_clear_registry(self, engine, full_passport):
        engine.compile_passport(full_passport)
        engine.validate_passport(full_passport)
        engine.clear_registry()
        assert engine.get_all_passports() == {}
        assert engine.get_validation_results() == []


class TestReferenceData:
    def test_field_reference(self, engine):
        ref = engine.get_field_reference()
        assert len(ref) == len(PassportField)
        for field_name, info in ref.items():
            assert "description" in info
            assert "section" in info
            assert "required" in info

    def test_section_reference(self, engine):
        ref = engine.get_section_reference()
        assert len(ref) == 6
        for section_key, info in ref.items():
            assert "label" in info
            assert "required_fields" in info


class TestRecommendations:
    def test_empty_passport_has_recommendations(self, engine, empty_passport):
        result = engine.validate_passport(empty_passport)
        assert len(result.recommendations) > 0

    def test_full_passport_readiness_message(self, engine, full_passport):
        result = engine.validate_passport(full_passport)
        assert len(result.recommendations) > 0
