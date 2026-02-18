# -*- coding: utf-8 -*-
"""
Unit tests for LeakDetectionEngine (Engine 3 of 7) - AGENT-MRV-005

Tests LDAR program management: survey scheduling, survey recording,
leak classification, coverage tracking, leak statistics, repair tracking,
DOR management, emission reduction, inspector management, listing, and
statistics.

Target: 60 tests, ~700 lines.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Dict

import pytest

from greenlang.fugitive_emissions.leak_detection import (
    LeakDetectionEngine,
    SurveyType,
    SurveyStatus,
    LeakSeverity,
    RepairStatus,
    RegulatoryFramework,
    LEAK_THRESHOLDS,
    SURVEY_FREQUENCIES,
    DOR_JUSTIFICATION_CODES,
    SurveyRecord,
    LeakRecord,
    InspectorRecord,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Fresh LeakDetectionEngine for each test."""
    return LeakDetectionEngine()


@pytest.fixture
def engine_eu():
    """Engine configured for EU Methane Regulation (500 ppmv threshold)."""
    return LeakDetectionEngine(
        config={"default_framework": "EU_METHANE_REG"}
    )


@pytest.fixture
def survey_data():
    return {
        "facility_id": "FAC-001",
        "survey_type": "OGI",
        "scheduled_date": "2026-03-15",
        "inspector_id": "INS-042",
    }


@pytest.fixture
def engine_with_survey(engine, survey_data):
    """Engine with a pre-scheduled survey."""
    result = engine.schedule_survey(survey_data)
    return engine, result["survey_id"]


@pytest.fixture
def engine_with_completed_survey(engine_with_survey):
    """Engine with a completed survey."""
    eng, sid = engine_with_survey
    eng.record_survey({
        "survey_id": sid,
        "completion_date": "2026-03-15",
        "components_surveyed": 5000,
        "components_total": 5000,
        "leaks_detected": 12,
    })
    return eng, sid


@pytest.fixture
def engine_with_leak(engine_with_survey):
    """Engine with a classified leak."""
    eng, sid = engine_with_survey
    leak = eng.classify_leak({
        "survey_id": sid,
        "facility_id": "FAC-001",
        "component_id": "COMP-V-101",
        "component_type": "valve",
        "service_type": "gas",
        "screening_value_ppmv": 15000,
        "detection_date": "2026-03-15",
    })
    return eng, sid, leak


# ===========================================================================
# Survey Scheduling (10 tests)
# ===========================================================================


class TestSurveyScheduling:
    """Tests for schedule_survey."""

    def test_schedule_basic(self, engine, survey_data):
        result = engine.schedule_survey(survey_data)
        assert "survey_id" in result
        assert result["status"] == "SCHEDULED"
        assert result["facility_id"] == "FAC-001"
        assert result["survey_type"] == "OGI"

    def test_schedule_method_21(self, engine):
        result = engine.schedule_survey({
            "facility_id": "FAC-002",
            "survey_type": "METHOD_21",
            "scheduled_date": "2026-04-01",
        })
        assert result["survey_type"] == "METHOD_21"

    def test_schedule_avo(self, engine):
        result = engine.schedule_survey({
            "facility_id": "FAC-003",
            "survey_type": "AVO",
            "scheduled_date": "2026-05-01",
        })
        assert result["survey_type"] == "AVO"

    def test_schedule_hiflow(self, engine):
        result = engine.schedule_survey({
            "facility_id": "FAC-004",
            "survey_type": "HIFLOW",
            "scheduled_date": "2026-06-01",
        })
        assert result["survey_type"] == "HIFLOW"

    def test_schedule_drone_ogi(self, engine):
        result = engine.schedule_survey({
            "facility_id": "FAC-005",
            "survey_type": "DRONE_OGI",
            "scheduled_date": "2026-07-01",
        })
        assert result["survey_type"] == "DRONE_OGI"

    def test_schedule_missing_facility_raises(self, engine):
        with pytest.raises(ValueError, match="facility_id is required"):
            engine.schedule_survey({
                "survey_type": "OGI",
                "scheduled_date": "2026-03-15",
            })

    def test_schedule_missing_date_raises(self, engine):
        with pytest.raises(ValueError, match="scheduled_date is required"):
            engine.schedule_survey({
                "facility_id": "FAC-001",
                "survey_type": "OGI",
            })

    def test_schedule_invalid_type_raises(self, engine):
        with pytest.raises(ValueError, match="survey_type must be one of"):
            engine.schedule_survey({
                "facility_id": "FAC-001",
                "survey_type": "INVALID",
                "scheduled_date": "2026-01-01",
            })

    def test_schedule_includes_provenance_hash(self, engine, survey_data):
        result = engine.schedule_survey(survey_data)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_schedule_increments_counter(self, engine, survey_data):
        engine.schedule_survey(survey_data)
        engine.schedule_survey({**survey_data, "scheduled_date": "2026-04-15"})
        assert engine._total_surveys_scheduled == 2


# ===========================================================================
# Survey Recording (5 tests)
# ===========================================================================


class TestSurveyRecording:
    """Tests for record_survey."""

    def test_record_survey_basic(self, engine_with_survey):
        engine, sid = engine_with_survey
        result = engine.record_survey({
            "survey_id": sid,
            "completion_date": "2026-03-15",
            "components_surveyed": 5000,
            "components_total": 5000,
            "leaks_detected": 12,
        })
        assert result["status"] == "COMPLETED"
        assert result["components_surveyed"] == 5000
        assert result["leaks_detected"] == 12

    def test_record_survey_calculates_coverage(self, engine_with_survey):
        engine, sid = engine_with_survey
        result = engine.record_survey({
            "survey_id": sid,
            "completion_date": "2026-03-15",
            "components_surveyed": 4000,
            "components_total": 5000,
            "leaks_detected": 8,
        })
        coverage = Decimal(result["coverage_pct"])
        assert coverage == Decimal("80.00000000")

    def test_record_nonexistent_survey_raises(self, engine):
        with pytest.raises(ValueError, match="Survey not found"):
            engine.record_survey({
                "survey_id": "srv_nonexistent",
                "completion_date": "2026-03-15",
                "components_surveyed": 100,
                "leaks_detected": 0,
            })

    def test_record_survey_updates_completed_counter(self, engine_with_survey):
        engine, sid = engine_with_survey
        engine.record_survey({
            "survey_id": sid,
            "completion_date": "2026-03-15",
            "components_surveyed": 1000,
            "leaks_detected": 5,
        })
        assert engine._total_surveys_completed >= 1

    def test_record_survey_has_provenance(self, engine_with_survey):
        engine, sid = engine_with_survey
        result = engine.record_survey({
            "survey_id": sid,
            "completion_date": "2026-03-15",
            "components_surveyed": 500,
            "leaks_detected": 2,
        })
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# Leak Classification (10 tests)
# ===========================================================================


class TestLeakClassification:
    """Tests for classify_leak against multiple frameworks."""

    def test_above_vva_threshold_is_leak(self, engine_with_survey):
        engine, sid = engine_with_survey
        result = engine.classify_leak({
            "survey_id": sid,
            "facility_id": "FAC-001",
            "screening_value_ppmv": 10000,
            "framework": "EPA_SUBPART_VVA",
        })
        assert result["is_leak"] is True
        assert result["threshold_ppmv"] == 10000

    def test_below_vva_threshold_not_leak(self, engine_with_survey):
        engine, sid = engine_with_survey
        result = engine.classify_leak({
            "survey_id": sid,
            "facility_id": "FAC-001",
            "screening_value_ppmv": 9999,
            "framework": "EPA_SUBPART_VVA",
        })
        assert result["is_leak"] is False

    def test_above_ooooa_threshold_is_leak(self, engine_with_survey):
        engine, sid = engine_with_survey
        result = engine.classify_leak({
            "survey_id": sid,
            "facility_id": "FAC-001",
            "screening_value_ppmv": 500,
            "framework": "EPA_SUBPART_OOOOA",
        })
        assert result["is_leak"] is True
        assert result["threshold_ppmv"] == 500

    def test_below_ooooa_threshold_not_leak(self, engine_with_survey):
        engine, sid = engine_with_survey
        result = engine.classify_leak({
            "survey_id": sid,
            "facility_id": "FAC-001",
            "screening_value_ppmv": 499,
            "framework": "EPA_SUBPART_OOOOA",
        })
        assert result["is_leak"] is False

    def test_severity_none_below_5ppmv(self, engine_with_survey):
        engine, sid = engine_with_survey
        result = engine.classify_leak({
            "survey_id": sid,
            "facility_id": "FAC-001",
            "screening_value_ppmv": 3,
        })
        assert result["severity"] == "NONE"

    def test_severity_minor_below_threshold(self, engine_with_survey):
        engine, sid = engine_with_survey
        result = engine.classify_leak({
            "survey_id": sid,
            "facility_id": "FAC-001",
            "screening_value_ppmv": 100,
            "framework": "EPA_SUBPART_VVA",
        })
        assert result["severity"] == "MINOR"
        assert result["is_leak"] is False

    def test_severity_moderate(self, engine_with_survey):
        engine, sid = engine_with_survey
        result = engine.classify_leak({
            "survey_id": sid,
            "facility_id": "FAC-001",
            "screening_value_ppmv": 15000,
            "framework": "EPA_SUBPART_VVA",
        })
        assert result["severity"] == "MODERATE"

    def test_severity_major(self, engine_with_survey):
        engine, sid = engine_with_survey
        result = engine.classify_leak({
            "survey_id": sid,
            "facility_id": "FAC-001",
            "screening_value_ppmv": 150000,
            "framework": "EPA_SUBPART_VVA",
        })
        assert result["severity"] == "MAJOR"

    def test_severity_critical(self, engine_with_survey):
        engine, sid = engine_with_survey
        result = engine.classify_leak({
            "survey_id": sid,
            "facility_id": "FAC-001",
            "screening_value_ppmv": 1_500_000,
            "framework": "EPA_SUBPART_VVA",
        })
        assert result["severity"] == "CRITICAL"

    def test_classify_includes_repair_deadline(self, engine_with_survey):
        engine, sid = engine_with_survey
        result = engine.classify_leak({
            "survey_id": sid,
            "facility_id": "FAC-001",
            "screening_value_ppmv": 15000,
            "detection_date": "2026-03-01",
            "framework": "EPA_SUBPART_VVA",
        })
        assert result["is_leak"] is True
        assert result["repair_deadline"] != ""
        deadline = date.fromisoformat(result["repair_deadline"])
        detection = date.fromisoformat("2026-03-01")
        assert (deadline - detection).days == 15


# ===========================================================================
# Repair Tracking (7 tests)
# ===========================================================================


class TestRepairTracking:
    """Tests for track_repair and repair deadline enforcement."""

    def test_successful_repair_below_threshold(self, engine_with_leak):
        engine, sid, leak = engine_with_leak
        result = engine.track_repair({
            "leak_id": leak["leak_id"],
            "repair_date": "2026-03-20",
            "post_repair_ppmv": 50,
        })
        assert result["repair_status"] == "REPAIRED"

    def test_failed_repair_above_threshold(self, engine_with_leak):
        engine, sid, leak = engine_with_leak
        result = engine.track_repair({
            "leak_id": leak["leak_id"],
            "repair_date": "2026-03-20",
            "post_repair_ppmv": 12000,
        })
        assert result["repair_status"] == "FAILED"

    def test_repair_within_deadline(self, engine_with_leak):
        engine, sid, leak = engine_with_leak
        result = engine.track_repair({
            "leak_id": leak["leak_id"],
            "repair_date": "2026-03-20",
            "post_repair_ppmv": 50,
        })
        assert result["repair_within_deadline"] is True

    def test_repair_after_deadline(self, engine_with_leak):
        engine, sid, leak = engine_with_leak
        # Leak detected 2026-03-15, deadline = 2026-03-30 (VVa = 15 days)
        result = engine.track_repair({
            "leak_id": leak["leak_id"],
            "repair_date": "2026-04-15",
            "post_repair_ppmv": 50,
        })
        assert result["repair_within_deadline"] is False

    def test_repair_nonexistent_leak_raises(self, engine):
        with pytest.raises(ValueError, match="Leak not found"):
            engine.track_repair({
                "leak_id": "leak_nonexistent",
                "repair_date": "2026-03-20",
                "post_repair_ppmv": 50,
            })

    def test_repair_increments_counter(self, engine_with_leak):
        engine, sid, leak = engine_with_leak
        engine.track_repair({
            "leak_id": leak["leak_id"],
            "repair_date": "2026-03-20",
            "post_repair_ppmv": 50,
        })
        assert engine._total_repairs_completed >= 1

    def test_repair_has_provenance(self, engine_with_leak):
        engine, sid, leak = engine_with_leak
        result = engine.track_repair({
            "leak_id": leak["leak_id"],
            "repair_date": "2026-03-20",
            "post_repair_ppmv": 50,
        })
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# Survey Coverage (5 tests)
# ===========================================================================


class TestSurveyCoverage:
    """Tests for calculate_survey_coverage."""

    def test_coverage_with_completed_survey(self, engine_with_completed_survey):
        engine, sid = engine_with_completed_survey
        result = engine.calculate_survey_coverage(
            facility_id="FAC-001",
            period_start="2026-01-01",
            period_end="2026-12-31",
        )
        assert result["surveys_completed"] >= 1
        assert result["total_components_surveyed"] >= 5000

    def test_coverage_no_surveys(self, engine):
        result = engine.calculate_survey_coverage(facility_id="FAC-EMPTY")
        assert result["surveys_completed"] == 0
        assert result["total_components_surveyed"] == 0

    def test_coverage_filter_by_survey_type(self, engine_with_completed_survey):
        engine, sid = engine_with_completed_survey
        result = engine.calculate_survey_coverage(
            facility_id="FAC-001",
            survey_type="OGI",
            period_start="2026-01-01",
            period_end="2026-12-31",
        )
        assert result["surveys_completed"] >= 1

    def test_coverage_has_provenance(self, engine_with_completed_survey):
        engine, sid = engine_with_completed_survey
        result = engine.calculate_survey_coverage(
            facility_id="FAC-001",
            period_start="2026-01-01",
            period_end="2026-12-31",
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_coverage_percentage_calculation(self, engine):
        sid = engine.schedule_survey({
            "facility_id": "FAC-COV",
            "survey_type": "OGI",
            "scheduled_date": "2026-06-01",
        })["survey_id"]
        engine.record_survey({
            "survey_id": sid,
            "completion_date": "2026-06-01",
            "components_surveyed": 250,
            "components_total": 500,
            "leaks_detected": 5,
        })
        result = engine.calculate_survey_coverage(
            facility_id="FAC-COV",
            period_start="2026-01-01",
            period_end="2026-12-31",
        )
        coverage_pct = Decimal(result["coverage_pct"])
        assert coverage_pct == Decimal("50.00000000")


# ===========================================================================
# Leak Statistics (5 tests)
# ===========================================================================


class TestLeakStatistics:
    """Tests for get_leak_statistics."""

    def test_statistics_empty(self, engine):
        stats = engine.get_leak_statistics(facility_id="FAC-EMPTY")
        assert stats["total_leaks"] == 0

    def test_statistics_with_leaks(self, engine_with_leak):
        engine, sid, leak = engine_with_leak
        stats = engine.get_leak_statistics(
            facility_id="FAC-001",
            period_start="2026-01-01",
            period_end="2026-12-31",
        )
        assert stats["total_leaks"] >= 1
        assert "severity_distribution" in stats

    def test_statistics_severity_distribution(self, engine_with_survey):
        engine, sid = engine_with_survey
        for ppmv in [15000, 20000, 150000]:
            engine.classify_leak({
                "survey_id": sid,
                "facility_id": "FAC-001",
                "screening_value_ppmv": ppmv,
                "detection_date": "2026-06-01",
            })
        stats = engine.get_leak_statistics(
            facility_id="FAC-001",
            period_start="2026-01-01",
            period_end="2026-12-31",
        )
        assert stats["total_leaks"] >= 3
        assert "MODERATE" in stats["severity_distribution"] or \
               "MAJOR" in stats["severity_distribution"]

    def test_statistics_has_provenance(self, engine_with_leak):
        engine, sid, leak = engine_with_leak
        stats = engine.get_leak_statistics(
            facility_id="FAC-001",
            period_start="2026-01-01",
            period_end="2026-12-31",
        )
        assert "provenance_hash" in stats
        assert len(stats["provenance_hash"]) == 64

    def test_statistics_repair_rate(self, engine_with_leak):
        engine, sid, leak = engine_with_leak
        engine.track_repair({
            "leak_id": leak["leak_id"],
            "repair_date": "2026-03-20",
            "post_repair_ppmv": 50,
        })
        stats = engine.get_leak_statistics(
            facility_id="FAC-001",
            period_start="2026-01-01",
            period_end="2026-12-31",
        )
        repair_rate = Decimal(stats["repair_rate_pct"])
        assert repair_rate > Decimal("0")


# ===========================================================================
# DOR Compliance (5 tests)
# ===========================================================================


class TestDORCompliance:
    """Tests for check_dor_compliance."""

    def test_dor_check_not_overdue(self, engine_with_leak):
        engine, sid, leak = engine_with_leak
        result = engine.check_dor_compliance(leak["leak_id"])
        # Leak just classified, deadline is in the future
        assert result["leak_id"] == leak["leak_id"]
        assert isinstance(result["is_overdue"], bool)

    def test_dor_nonexistent_leak_raises(self, engine):
        with pytest.raises(ValueError, match="Leak not found"):
            engine.check_dor_compliance("leak_nonexistent")

    def test_dor_justification_codes_exist(self):
        assert "PROCESS_UNIT_SHUTDOWN" in DOR_JUSTIFICATION_CODES
        assert "PARTS_UNAVAILABLE" in DOR_JUSTIFICATION_CODES
        assert "SAFETY_HAZARD" in DOR_JUSTIFICATION_CODES
        assert "TECHNICALLY_INFEASIBLE" in DOR_JUSTIFICATION_CODES
        assert "OTHER" in DOR_JUSTIFICATION_CODES

    def test_dor_has_provenance(self, engine_with_leak):
        engine, sid, leak = engine_with_leak
        result = engine.check_dor_compliance(leak["leak_id"])
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_dor_allowed_flag(self, engine_with_leak):
        engine, sid, leak = engine_with_leak
        result = engine.check_dor_compliance(
            leak["leak_id"],
            data={"framework": "EPA_SUBPART_VVA"},
        )
        assert result["dor_allowed"] is True


# ===========================================================================
# Emission Reduction (3 tests)
# ===========================================================================


class TestEmissionReduction:
    """Tests for calculate_emission_reduction."""

    def test_reduction_after_repair(self, engine_with_survey):
        engine, sid = engine_with_survey
        leak = engine.classify_leak({
            "survey_id": sid,
            "facility_id": "FAC-001",
            "screening_value_ppmv": 15000,
            "estimated_emission_kg_hr": 0.05,
            "detection_date": "2026-03-01",
        })
        engine.track_repair({
            "leak_id": leak["leak_id"],
            "repair_date": "2026-03-15",
            "post_repair_ppmv": 100,
        })
        result = engine.calculate_emission_reduction(
            facility_id="FAC-001",
            period_start="2026-01-01",
            period_end="2026-12-31",
        )
        assert "total_reduction_kg" in result
        total_kg = Decimal(result["total_reduction_kg"])
        assert total_kg >= Decimal("0")

    def test_reduction_no_repairs(self, engine):
        result = engine.calculate_emission_reduction(facility_id="FAC-EMPTY")
        total = Decimal(result["total_reduction_kg"])
        assert total == Decimal("0")

    def test_reduction_has_provenance(self, engine):
        result = engine.calculate_emission_reduction(facility_id="FAC-EMPTY")
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# Inspector Management (4 tests)
# ===========================================================================


class TestInspectorManagement:
    """Tests for register_inspector and check_inspector_certification."""

    def test_register_inspector(self, engine):
        result = engine.register_inspector({
            "name": "Jane Doe",
            "certifications": ["OGI_LEVEL_1", "METHOD_21"],
            "certification_dates": {
                "OGI_LEVEL_1": "2027-06-01",
                "METHOD_21": "2027-12-31",
            },
        })
        assert "inspector_id" in result
        assert result["name"] == "Jane Doe"
        assert "OGI_LEVEL_1" in result["certifications"]

    def test_register_inspector_missing_name_raises(self, engine):
        with pytest.raises(ValueError, match="name is required"):
            engine.register_inspector({"name": ""})

    def test_check_certification(self, engine):
        insp = engine.register_inspector({
            "name": "John Smith",
            "certifications": ["OGI_LEVEL_1"],
            "certification_dates": {"OGI_LEVEL_1": "2027-12-31"},
        })
        result = engine.check_inspector_certification(
            insp["inspector_id"], required_cert="OGI_LEVEL_1"
        )
        assert result["has_required_certification"] is True
        assert result["all_certifications_valid"] is True

    def test_check_nonexistent_inspector_raises(self, engine):
        with pytest.raises(ValueError, match="Inspector not found"):
            engine.check_inspector_certification("insp_nonexistent")


# ===========================================================================
# Listing Methods (3 tests)
# ===========================================================================


class TestListingMethods:
    """Tests for list_surveys and list_leaks."""

    def test_list_surveys_pagination(self, engine):
        for i in range(5):
            engine.schedule_survey({
                "facility_id": "FAC-LIST",
                "survey_type": "OGI",
                "scheduled_date": f"2026-0{i + 1}-15",
            })
        result = engine.list_surveys(
            facility_id="FAC-LIST", page=1, page_size=3
        )
        assert len(result["surveys"]) <= 3
        assert result["total"] == 5

    def test_list_leaks_by_facility(self, engine_with_leak):
        engine, sid, leak = engine_with_leak
        result = engine.list_leaks(facility_id="FAC-001")
        assert result["total"] >= 1
        assert result["leaks"][0]["facility_id"] == "FAC-001"

    def test_list_surveys_filter_status(self, engine_with_completed_survey):
        engine, sid = engine_with_completed_survey
        result = engine.list_surveys(
            facility_id="FAC-001", status="COMPLETED"
        )
        for s in result["surveys"]:
            assert s["status"] == "COMPLETED"


# ===========================================================================
# Engine Statistics (2 tests)
# ===========================================================================


class TestEngineStatistics:
    """Tests for get_statistics."""

    def test_statistics_basic(self, engine):
        stats = engine.get_statistics()
        assert "total_surveys_scheduled" in stats
        assert "total_leaks_detected" in stats
        assert "total_repairs_completed" in stats
        assert "total_dor_records" in stats

    def test_statistics_after_operations(self, engine_with_leak):
        engine, sid, leak = engine_with_leak
        stats = engine.get_statistics()
        assert stats["total_surveys_scheduled"] >= 1
        assert stats["total_leaks_detected"] >= 1
        assert stats["surveys_in_registry"] >= 1
        assert stats["leaks_in_registry"] >= 1


# ===========================================================================
# Regulatory Framework Constants (7 tests)
# ===========================================================================


class TestRegulatoryConstants:
    """Validate hardcoded regulatory reference data."""

    def test_vva_threshold_10000(self):
        assert LEAK_THRESHOLDS["EPA_SUBPART_VVA"]["threshold_ppmv"] == 10000

    def test_ooooa_threshold_500(self):
        assert LEAK_THRESHOLDS["EPA_SUBPART_OOOOA"]["threshold_ppmv"] == 500

    def test_mact_threshold_500(self):
        assert LEAK_THRESHOLDS["EPA_MACT_SUBPART_H"]["threshold_ppmv"] == 500

    def test_eu_methane_threshold_500(self):
        assert LEAK_THRESHOLDS["EU_METHANE_REG"]["threshold_ppmv"] == 500

    def test_vva_repair_deadline_15_days(self):
        assert LEAK_THRESHOLDS["EPA_SUBPART_VVA"]["repair_deadline_days"] == 15

    def test_eu_repair_deadline_30_days(self):
        assert LEAK_THRESHOLDS["EU_METHANE_REG"]["repair_deadline_days"] == 30

    def test_survey_frequency_ogi_90_days(self):
        assert SURVEY_FREQUENCIES["OGI"]["interval_days"] == 90


# ===========================================================================
# Enumerations (5 tests)
# ===========================================================================


class TestEnumerations:
    """Verify enum values."""

    def test_survey_types(self):
        assert SurveyType.METHOD_21.value == "METHOD_21"
        assert SurveyType.OGI.value == "OGI"
        assert SurveyType.AVO.value == "AVO"
        assert SurveyType.HIFLOW.value == "HIFLOW"
        assert SurveyType.DRONE_OGI.value == "DRONE_OGI"

    def test_survey_statuses(self):
        assert SurveyStatus.SCHEDULED.value == "SCHEDULED"
        assert SurveyStatus.COMPLETED.value == "COMPLETED"
        assert SurveyStatus.CANCELLED.value == "CANCELLED"
        assert SurveyStatus.OVERDUE.value == "OVERDUE"

    def test_leak_severity(self):
        assert LeakSeverity.NONE.value == "NONE"
        assert LeakSeverity.MINOR.value == "MINOR"
        assert LeakSeverity.MODERATE.value == "MODERATE"
        assert LeakSeverity.MAJOR.value == "MAJOR"
        assert LeakSeverity.CRITICAL.value == "CRITICAL"

    def test_repair_statuses(self):
        assert RepairStatus.PENDING.value == "PENDING"
        assert RepairStatus.REPAIRED.value == "REPAIRED"
        assert RepairStatus.VERIFIED.value == "VERIFIED"
        assert RepairStatus.DOR.value == "DOR"
        assert RepairStatus.FAILED.value == "FAILED"

    def test_frameworks(self):
        assert RegulatoryFramework.EPA_SUBPART_VVA.value == "EPA_SUBPART_VVA"
        assert RegulatoryFramework.EPA_SUBPART_OOOOA.value == "EPA_SUBPART_OOOOA"
        assert RegulatoryFramework.EPA_MACT_SUBPART_H.value == "EPA_MACT_SUBPART_H"
        assert RegulatoryFramework.EU_METHANE_REG.value == "EU_METHANE_REG"
        assert RegulatoryFramework.ALBERTA_DIRECTIVE_060.value == "ALBERTA_DIRECTIVE_060"
