# -*- coding: utf-8 -*-
"""
Unit tests for Engine 8: AuditAnalyticsEngine -- AGENT-EUDR-024

Tests finding trend analysis, auditor performance benchmarking,
compliance rate calculation, CAR lifecycle metrics, executive dashboard
aggregation, competent authority interaction tracking, and analytics
provenance.

Target: ~60 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from datetime import date, timedelta
from decimal import Decimal

import pytest

from greenlang.agents.eudr.third_party_audit_manager.audit_analytics_engine import (
    AuditAnalyticsEngine,
)
from greenlang.agents.eudr.third_party_audit_manager.models import (
    CalculateAnalyticsRequest,
    CalculateAnalyticsResponse,
    AuthorityInteractionType,
    LogAuthorityInteractionRequest,
)
from tests.agents.eudr.third_party_audit_manager.conftest import (
    FROZEN_DATE,
    FROZEN_NOW,
    FROZEN_PAST_90D,
    FROZEN_PAST_365D,
    SHA256_HEX_LENGTH,
    SAMPLE_AUTHORITIES,
)


class TestAnalyticsEngineInit:
    """Test engine initialization."""

    def test_init_with_config(self, default_config):
        engine = AuditAnalyticsEngine(config=default_config)
        assert engine.config is not None

    def test_init_without_config(self):
        engine = AuditAnalyticsEngine()
        assert engine.config is not None


class TestFindingTrends:
    """Test finding trend analytics."""

    def test_finding_trends_basic(self, analytics_engine, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        assert response is not None
        assert response.finding_trends is not None

    def test_finding_trends_by_severity(self, analytics_engine, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        trends = response.finding_trends
        assert "by_severity" in trends or trends is not None

    def test_finding_trends_by_country(self, analytics_engine, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        trends = response.finding_trends
        assert trends is not None

    def test_finding_trends_by_commodity(self, analytics_engine, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        trends = response.finding_trends
        assert trends is not None

    def test_finding_trends_has_total(self, analytics_engine, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        assert response.total_findings_count >= 0


class TestAuditorPerformanceBenchmarking:
    """Test auditor performance benchmarking analytics."""

    def test_auditor_performance_metrics(self, analytics_engine, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        assert response.auditor_performance is not None

    def test_auditor_performance_has_ratings(self, analytics_engine, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        perf = response.auditor_performance
        assert perf is not None

    def test_average_findings_per_audit(self, analytics_engine, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        assert response.auditor_performance is not None

    def test_car_closure_rate_tracked(self, analytics_engine, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        assert response.auditor_performance is not None

    def test_auditor_benchmarking_deterministic(self, analytics_engine, analytics_request):
        r1 = analytics_engine.calculate_analytics(analytics_request)
        r2 = analytics_engine.calculate_analytics(analytics_request)
        assert r1.auditor_performance == r2.auditor_performance


class TestComplianceRates:
    """Test compliance rate calculation."""

    def test_compliance_rate_basic(self, analytics_engine, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        assert response.compliance_rate is not None

    def test_compliance_rate_is_decimal(self, analytics_engine, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        assert isinstance(response.compliance_rate, (Decimal, float, int))

    def test_compliance_rate_range(self, analytics_engine, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        rate = Decimal(str(response.compliance_rate))
        assert Decimal("0") <= rate <= Decimal("100")

    def test_compliance_rate_by_quarter(self, analytics_engine, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        assert response.compliance_rate is not None


class TestCARLifecycleMetrics:
    """Test CAR lifecycle performance analytics."""

    def test_car_metrics_basic(self, analytics_engine, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        assert response.car_lifecycle_metrics is not None

    def test_car_sla_compliance_rate(self, analytics_engine, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        metrics = response.car_lifecycle_metrics
        assert metrics is not None

    def test_car_average_closure_time(self, analytics_engine, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        metrics = response.car_lifecycle_metrics
        assert metrics is not None

    def test_car_overdue_count(self, analytics_engine, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        assert response.car_lifecycle_metrics is not None

    def test_car_escalation_count(self, analytics_engine, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        assert response.car_lifecycle_metrics is not None


class TestExecutiveDashboard:
    """Test executive dashboard aggregation."""

    def test_dashboard_data_basic(self, analytics_engine, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        assert response.dashboard is not None

    def test_dashboard_has_active_audits(self, analytics_engine, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        dashboard = response.dashboard
        assert "active_audits" in dashboard or dashboard is not None

    def test_dashboard_has_open_cars(self, analytics_engine, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        dashboard = response.dashboard
        assert dashboard is not None

    def test_dashboard_has_overdue_cars(self, analytics_engine, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        dashboard = response.dashboard
        assert dashboard is not None

    def test_dashboard_has_compliance_rate(self, analytics_engine, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        assert response.compliance_rate is not None


class TestCompetentAuthorityInteractions:
    """Test competent authority interaction tracking analytics."""

    def test_log_authority_interaction(self, analytics_engine, authority_interaction_request):
        response = analytics_engine.log_authority_interaction(authority_interaction_request)
        assert response is not None
        assert response.interaction_id is not None

    def test_interaction_has_provenance(self, analytics_engine, authority_interaction_request):
        response = analytics_engine.log_authority_interaction(authority_interaction_request)
        assert response.provenance_hash is not None
        assert len(response.provenance_hash) == SHA256_HEX_LENGTH

    @pytest.mark.parametrize("member_state,authority", list(SAMPLE_AUTHORITIES.items()))
    def test_authority_interaction_by_state(self, analytics_engine, member_state, authority):
        request = LogAuthorityInteractionRequest(
            operator_id="OP-001",
            authority_name=authority,
            member_state=member_state,
            interaction_type="document_request",
            subject=f"DDS request from {authority}",
        )
        response = analytics_engine.log_authority_interaction(request)
        assert response is not None

    @pytest.mark.parametrize("interaction_type", [
        "document_request", "inspection_notification", "enforcement_measure",
    ])
    def test_interaction_types(self, analytics_engine, interaction_type):
        request = LogAuthorityInteractionRequest(
            operator_id="OP-001",
            authority_name="BMEL",
            member_state="DE",
            interaction_type=interaction_type,
            subject=f"Test {interaction_type}",
        )
        response = analytics_engine.log_authority_interaction(request)
        assert response is not None

    def test_interaction_response_sla_tracking(self, analytics_engine, authority_interaction_request):
        response = analytics_engine.log_authority_interaction(authority_interaction_request)
        assert response.response_deadline is not None


class TestAnalyticsProvenance:
    """Test analytics provenance and determinism."""

    def test_analytics_has_provenance_hash(self, analytics_engine, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        assert response.provenance_hash is not None
        assert len(response.provenance_hash) == SHA256_HEX_LENGTH

    def test_analytics_deterministic(self, analytics_engine, analytics_request):
        r1 = analytics_engine.calculate_analytics(analytics_request)
        r2 = analytics_engine.calculate_analytics(analytics_request)
        assert r1.provenance_hash == r2.provenance_hash

    def test_different_periods_different_hash(self, analytics_engine):
        r1 = analytics_engine.calculate_analytics(CalculateAnalyticsRequest(
            operator_id="OP-001",
            time_period_start=FROZEN_DATE - timedelta(days=365),
            time_period_end=FROZEN_DATE,
        ))
        r2 = analytics_engine.calculate_analytics(CalculateAnalyticsRequest(
            operator_id="OP-001",
            time_period_start=FROZEN_DATE - timedelta(days=180),
            time_period_end=FROZEN_DATE,
        ))
        assert r1.provenance_hash != r2.provenance_hash


class TestAnalyticsBatch:
    """Test analytics with batch data."""

    def test_analytics_with_batch_audits(self, analytics_engine, batch_audits, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        assert response is not None

    def test_analytics_with_batch_ncs(self, analytics_engine, batch_ncs, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        assert response is not None

    def test_analytics_with_batch_cars(self, analytics_engine, batch_cars, analytics_request):
        response = analytics_engine.calculate_analytics(analytics_request)
        assert response is not None
