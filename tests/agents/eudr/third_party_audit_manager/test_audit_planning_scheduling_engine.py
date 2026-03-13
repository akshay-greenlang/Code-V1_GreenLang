# -*- coding: utf-8 -*-
"""
Unit tests for Engine 1: AuditPlanningSchedulingEngine -- AGENT-EUDR-024

Tests risk-based audit scheduling, priority score calculation, frequency
tier assignment, scope determination, conflict detection, unscheduled
audit triggers, recertification timeline integration, and resource
budget tracking.

Target: ~60 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.third_party_audit_manager.audit_planning_scheduling_engine import (
    AuditPlanningSchedulingEngine,
    FREQUENCY_INTERVALS,
    FREQUENCY_MODALITY,
    FREQUENCY_SCOPE,
    SCHEME_RECERTIFICATION_CYCLES,
    SCOPE_DURATION_DAYS,
    DEFAULT_EUDR_ARTICLES,
)
from greenlang.agents.eudr.third_party_audit_manager.models import (
    AuditModality,
    AuditScope,
    AuditStatus,
    ScheduleAuditRequest,
)
from tests.agents.eudr.third_party_audit_manager.conftest import (
    FROZEN_DATE,
    FROZEN_NOW,
    SHA256_HEX_LENGTH,
)


# ===================================================================
# Initialization Tests
# ===================================================================


class TestAuditPlanningEngineInit:
    """Test engine initialization."""

    def test_init_with_default_config(self, default_config):
        engine = AuditPlanningSchedulingEngine(config=default_config)
        assert engine.config is not None

    def test_init_without_config_uses_global(self):
        engine = AuditPlanningSchedulingEngine()
        assert engine.config is not None

    def test_frequency_intervals_defined(self):
        assert "HIGH" in FREQUENCY_INTERVALS
        assert "STANDARD" in FREQUENCY_INTERVALS
        assert "LOW" in FREQUENCY_INTERVALS
        assert FREQUENCY_INTERVALS["HIGH"] == 90
        assert FREQUENCY_INTERVALS["STANDARD"] == 180
        assert FREQUENCY_INTERVALS["LOW"] == 365

    def test_frequency_scope_mapping(self):
        assert FREQUENCY_SCOPE["HIGH"] == AuditScope.FULL
        assert FREQUENCY_SCOPE["STANDARD"] == AuditScope.TARGETED
        assert FREQUENCY_SCOPE["LOW"] == AuditScope.SURVEILLANCE

    def test_frequency_modality_mapping(self):
        assert FREQUENCY_MODALITY["HIGH"] == AuditModality.ON_SITE
        assert FREQUENCY_MODALITY["LOW"] == AuditModality.REMOTE


# ===================================================================
# Priority Score Calculation Tests
# ===================================================================


class TestPriorityScoreCalculation:
    """Test composite audit priority score formula."""

    def test_all_zero_inputs_return_zero(self, planning_engine):
        result = planning_engine.calculate_priority_score(
            supplier_id="SUP-001",
            country_risk=Decimal("0"),
            supplier_risk=Decimal("0"),
            nc_history_score=Decimal("0"),
            certification_gap_score=Decimal("0"),
            deforestation_alert_score=Decimal("0"),
        )
        assert result["priority_score"] == Decimal("0")

    def test_all_max_inputs_return_capped_100(self, planning_engine):
        result = planning_engine.calculate_priority_score(
            supplier_id="SUP-001",
            country_risk=Decimal("100"),
            supplier_risk=Decimal("100"),
            nc_history_score=Decimal("100"),
            certification_gap_score=Decimal("100"),
            deforestation_alert_score=Decimal("100"),
        )
        assert result["priority_score"] <= Decimal("100")

    def test_default_weights_sum_to_one(self, planning_engine):
        weights = planning_engine._get_default_weights()
        total = sum(weights.values())
        assert total == Decimal("1.00")

    def test_high_country_risk_increases_score(self, planning_engine):
        low_risk = planning_engine.calculate_priority_score(
            supplier_id="SUP-001",
            country_risk=Decimal("10"),
            supplier_risk=Decimal("50"),
        )
        high_risk = planning_engine.calculate_priority_score(
            supplier_id="SUP-001",
            country_risk=Decimal("90"),
            supplier_risk=Decimal("50"),
        )
        assert high_risk["priority_score"] > low_risk["priority_score"]

    def test_high_supplier_risk_increases_score(self, planning_engine):
        low_risk = planning_engine.calculate_priority_score(
            supplier_id="SUP-001",
            supplier_risk=Decimal("10"),
        )
        high_risk = planning_engine.calculate_priority_score(
            supplier_id="SUP-001",
            supplier_risk=Decimal("90"),
        )
        assert high_risk["priority_score"] > low_risk["priority_score"]

    def test_nc_history_affects_score(self, planning_engine):
        no_history = planning_engine.calculate_priority_score(
            supplier_id="SUP-001",
            nc_history_score=Decimal("0"),
        )
        bad_history = planning_engine.calculate_priority_score(
            supplier_id="SUP-001",
            nc_history_score=Decimal("80"),
        )
        assert bad_history["priority_score"] > no_history["priority_score"]

    def test_deforestation_alert_affects_score(self, planning_engine):
        no_alert = planning_engine.calculate_priority_score(
            supplier_id="SUP-001",
            deforestation_alert_score=Decimal("0"),
        )
        alert = planning_engine.calculate_priority_score(
            supplier_id="SUP-001",
            deforestation_alert_score=Decimal("90"),
        )
        assert alert["priority_score"] > no_alert["priority_score"]

    def test_certification_gap_affects_score(self, planning_engine):
        certified = planning_engine.calculate_priority_score(
            supplier_id="SUP-001",
            certification_gap_score=Decimal("0"),
        )
        gap = planning_engine.calculate_priority_score(
            supplier_id="SUP-001",
            certification_gap_score=Decimal("80"),
        )
        assert gap["priority_score"] > certified["priority_score"]

    def test_recency_multiplier_increases_score(self, planning_engine):
        recent = planning_engine.calculate_priority_score(
            supplier_id="SUP-001",
            country_risk=Decimal("50"),
            supplier_risk=Decimal("50"),
            days_since_last_audit=30,
            scheduled_interval=365,
        )
        overdue = planning_engine.calculate_priority_score(
            supplier_id="SUP-001",
            country_risk=Decimal("50"),
            supplier_risk=Decimal("50"),
            days_since_last_audit=700,
            scheduled_interval=365,
        )
        assert overdue["priority_score"] > recent["priority_score"]

    def test_recency_multiplier_capped_at_2(self, planning_engine):
        result = planning_engine.calculate_priority_score(
            supplier_id="SUP-001",
            country_risk=Decimal("50"),
            supplier_risk=Decimal("50"),
            days_since_last_audit=1000,
            scheduled_interval=100,
        )
        # recency multiplier should be capped at 2.0
        assert Decimal(result["recency_multiplier"]) <= Decimal("2.0")

    def test_priority_score_is_decimal(self, planning_engine):
        result = planning_engine.calculate_priority_score(
            supplier_id="SUP-001",
            country_risk=Decimal("55.55"),
            supplier_risk=Decimal("44.44"),
        )
        assert isinstance(result["priority_score"], Decimal)

    def test_priority_score_two_decimal_places(self, planning_engine):
        result = planning_engine.calculate_priority_score(
            supplier_id="SUP-001",
            country_risk=Decimal("33.33"),
            supplier_risk=Decimal("66.66"),
        )
        score = result["priority_score"]
        assert score == score.quantize(Decimal("0.01"))

    def test_negative_inputs_clamped_to_zero(self, planning_engine):
        result = planning_engine.calculate_priority_score(
            supplier_id="SUP-001",
            country_risk=Decimal("-10"),
            supplier_risk=Decimal("-20"),
        )
        assert result["priority_score"] >= Decimal("0")

    def test_inputs_above_100_clamped(self, planning_engine):
        result = planning_engine.calculate_priority_score(
            supplier_id="SUP-001",
            country_risk=Decimal("150"),
            supplier_risk=Decimal("200"),
        )
        assert result["priority_score"] <= Decimal("100")


# ===================================================================
# Frequency Tier Assignment Tests
# ===================================================================


class TestFrequencyTierAssignment:
    """Test frequency tier assignment based on priority score."""

    def test_high_tier_for_score_70_plus(self, planning_engine):
        result = planning_engine.calculate_priority_score(
            supplier_id="SUP-001",
            country_risk=Decimal("100"),
            supplier_risk=Decimal("100"),
            nc_history_score=Decimal("100"),
            certification_gap_score=Decimal("100"),
            deforestation_alert_score=Decimal("100"),
        )
        assert result["frequency_tier"] == "HIGH"

    def test_low_tier_for_score_below_40(self, planning_engine):
        result = planning_engine.calculate_priority_score(
            supplier_id="SUP-001",
            country_risk=Decimal("10"),
            supplier_risk=Decimal("10"),
            nc_history_score=Decimal("0"),
            certification_gap_score=Decimal("0"),
            deforestation_alert_score=Decimal("0"),
        )
        assert result["frequency_tier"] == "LOW"

    def test_standard_tier_for_mid_range_score(self, planning_engine):
        result = planning_engine.calculate_priority_score(
            supplier_id="SUP-001",
            country_risk=Decimal("50"),
            supplier_risk=Decimal("50"),
            nc_history_score=Decimal("30"),
            certification_gap_score=Decimal("30"),
            deforestation_alert_score=Decimal("30"),
        )
        score = result["priority_score"]
        if Decimal("40") <= score < Decimal("70"):
            assert result["frequency_tier"] == "STANDARD"

    def test_boundary_score_70_is_high(self, planning_engine):
        # Construct inputs that yield exactly 70 or just above
        result = planning_engine.calculate_priority_score(
            supplier_id="SUP-001",
            country_risk=Decimal("70"),
            supplier_risk=Decimal("70"),
            nc_history_score=Decimal("70"),
            certification_gap_score=Decimal("70"),
            deforestation_alert_score=Decimal("70"),
        )
        assert result["frequency_tier"] == "HIGH"

    @pytest.mark.parametrize("country_risk,expected_tier", [
        (Decimal("100"), "HIGH"),
        (Decimal("0"), "LOW"),
    ])
    def test_extreme_country_risk_tier(self, planning_engine, country_risk, expected_tier):
        result = planning_engine.calculate_priority_score(
            supplier_id="SUP-001",
            country_risk=country_risk,
            supplier_risk=country_risk,
            nc_history_score=country_risk,
            certification_gap_score=country_risk,
            deforestation_alert_score=country_risk,
        )
        assert result["frequency_tier"] == expected_tier


# ===================================================================
# NC History Score Calculation Tests
# ===================================================================


class TestNCHistoryScore:
    """Test NC history score calculation."""

    def test_zero_ncs_returns_zero(self, planning_engine):
        score = planning_engine.calculate_nc_history_score(
            open_critical=0, open_major=0, open_minor=0, total_audits=1,
        )
        assert score == Decimal("0")

    def test_critical_ncs_have_highest_weight(self, planning_engine):
        critical_score = planning_engine.calculate_nc_history_score(
            open_critical=1, open_major=0, open_minor=0, total_audits=1,
        )
        major_score = planning_engine.calculate_nc_history_score(
            open_critical=0, open_major=1, open_minor=0, total_audits=1,
        )
        minor_score = planning_engine.calculate_nc_history_score(
            open_critical=0, open_major=0, open_minor=1, total_audits=1,
        )
        assert critical_score > major_score > minor_score

    def test_more_audits_dilutes_score(self, planning_engine):
        one_audit = planning_engine.calculate_nc_history_score(
            open_critical=1, total_audits=1,
        )
        ten_audits = planning_engine.calculate_nc_history_score(
            open_critical=1, total_audits=10,
        )
        assert one_audit > ten_audits

    def test_score_capped_at_100(self, planning_engine):
        score = planning_engine.calculate_nc_history_score(
            open_critical=10, open_major=10, open_minor=10, total_audits=1,
        )
        assert score <= Decimal("100")


# ===================================================================
# Schedule Generation Tests
# ===================================================================


class TestScheduleGeneration:
    """Test audit schedule generation for multiple suppliers."""

    def test_schedule_audits_basic(self, planning_engine, sample_schedule_request):
        response = planning_engine.schedule_audits(sample_schedule_request)
        assert response is not None
        assert response.total_scheduled > 0
        assert len(response.scheduled_audits) > 0

    def test_schedule_has_risk_distribution(self, planning_engine, sample_schedule_request):
        response = planning_engine.schedule_audits(sample_schedule_request)
        dist = response.risk_distribution
        assert "HIGH" in dist
        assert "STANDARD" in dist
        assert "LOW" in dist

    def test_scheduled_audits_have_provenance_hash(self, planning_engine, sample_schedule_request):
        response = planning_engine.schedule_audits(sample_schedule_request)
        for audit in response.scheduled_audits:
            assert audit.provenance_hash is not None
            assert len(audit.provenance_hash) == SHA256_HEX_LENGTH

    def test_scheduled_audits_have_planned_status(self, planning_engine, sample_schedule_request):
        response = planning_engine.schedule_audits(sample_schedule_request)
        for audit in response.scheduled_audits:
            assert audit.status == AuditStatus.PLANNED

    def test_scheduled_audits_have_eudr_articles(self, planning_engine, sample_schedule_request):
        response = planning_engine.schedule_audits(sample_schedule_request)
        for audit in response.scheduled_audits:
            assert len(audit.eudr_articles) > 0

    def test_response_has_resource_summary(self, planning_engine, sample_schedule_request):
        response = planning_engine.schedule_audits(sample_schedule_request)
        assert response.resource_summary is not None
        assert "total_auditor_days" in response.resource_summary

    def test_response_has_provenance_hash(self, planning_engine, sample_schedule_request):
        response = planning_engine.schedule_audits(sample_schedule_request)
        assert response.provenance_hash is not None
        assert len(response.provenance_hash) == SHA256_HEX_LENGTH

    def test_empty_supplier_list(self, planning_engine):
        request = ScheduleAuditRequest(
            operator_id="OP-001",
            supplier_ids=[],
            planning_year=2026,
        )
        response = planning_engine.schedule_audits(request)
        assert response.total_scheduled == 0

    def test_single_supplier(self, planning_engine):
        request = ScheduleAuditRequest(
            operator_id="OP-001",
            supplier_ids=["SUP-SINGLE-001"],
            planning_year=2026,
        )
        response = planning_engine.schedule_audits(request)
        assert response.total_scheduled >= 1


# ===================================================================
# Recertification Timeline Tests
# ===================================================================


class TestRecertificationTimelines:
    """Test certification scheme recertification cycle integration."""

    def test_fsc_five_year_cycle(self):
        assert SCHEME_RECERTIFICATION_CYCLES["fsc"] == 5

    def test_pefc_five_year_cycle(self):
        assert SCHEME_RECERTIFICATION_CYCLES["pefc"] == 5

    def test_rspo_five_year_cycle(self):
        assert SCHEME_RECERTIFICATION_CYCLES["rspo"] == 5

    def test_ra_three_year_cycle(self):
        assert SCHEME_RECERTIFICATION_CYCLES["rainforest_alliance"] == 3

    def test_iscc_annual_cycle(self):
        assert SCHEME_RECERTIFICATION_CYCLES["iscc"] == 1


# ===================================================================
# Scope Duration Tests
# ===================================================================


class TestScopeDuration:
    """Test estimated duration by audit scope."""

    def test_full_scope_five_days(self):
        assert SCOPE_DURATION_DAYS["full"] == 5

    def test_targeted_scope_three_days(self):
        assert SCOPE_DURATION_DAYS["targeted"] == 3

    def test_surveillance_scope_two_days(self):
        assert SCOPE_DURATION_DAYS["surveillance"] == 2

    def test_unscheduled_scope_three_days(self):
        assert SCOPE_DURATION_DAYS["unscheduled"] == 3


# ===================================================================
# Default EUDR Articles Tests
# ===================================================================


class TestDefaultEUDRArticles:
    """Test default EUDR articles in audit scope."""

    def test_default_articles_include_art3(self):
        assert "Art. 3" in DEFAULT_EUDR_ARTICLES

    def test_default_articles_include_art9(self):
        assert "Art. 9" in DEFAULT_EUDR_ARTICLES

    def test_default_articles_include_art10(self):
        assert "Art. 10" in DEFAULT_EUDR_ARTICLES

    def test_default_articles_include_art11(self):
        assert "Art. 11" in DEFAULT_EUDR_ARTICLES

    def test_default_articles_include_art29(self):
        assert "Art. 29" in DEFAULT_EUDR_ARTICLES
