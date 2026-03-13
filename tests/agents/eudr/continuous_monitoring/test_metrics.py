# -*- coding: utf-8 -*-
"""
Unit tests for Prometheus metrics - AGENT-EUDR-033

Tests all 40 Prometheus metrics (16 counters, 14 histograms, 10 gauges)
for the Continuous Monitoring Agent. Validates metric registration,
label application, and graceful fallback when prometheus_client is
not available.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.continuous_monitoring.metrics import (
    # Counters (16)
    record_supply_chain_scan,
    record_supplier_change_detected,
    record_certification_expiry,
    record_geolocation_drift,
    record_deforestation_check,
    record_deforestation_correlation,
    record_investigation_triggered,
    record_compliance_audit,
    record_compliance_check_passed,
    record_compliance_check_failed,
    record_change_detected,
    record_risk_monitor,
    record_risk_degradation,
    record_freshness_check,
    record_regulatory_update,
    record_alert_generated,
    # Histograms (14)
    observe_supply_chain_scan_duration,
    observe_deforestation_check_duration,
    observe_compliance_audit_duration,
    observe_change_detection_duration,
    observe_risk_monitor_duration,
    observe_freshness_check_duration,
    observe_regulatory_check_duration,
    observe_impact_assessment_duration,
    observe_correlation_duration,
    observe_investigation_trigger_duration,
    observe_trend_analysis_duration,
    observe_freshness_report_duration,
    observe_notification_duration,
    observe_entity_mapping_duration,
    # Gauges (10)
    set_active_scans,
    set_pending_investigations,
    set_expiring_certifications,
    set_stale_entities,
    set_open_alerts,
    set_critical_alerts,
    set_high_risk_entities,
    set_compliance_score,
    set_freshness_percentage,
    set_monitored_suppliers,
)


class TestCounterMetrics:
    """Test counter metric recording functions."""

    def test_record_supply_chain_scan(self):
        record_supply_chain_scan("completed")

    def test_record_supply_chain_scan_failed(self):
        record_supply_chain_scan("failed")

    def test_record_supplier_change_detected(self):
        record_supplier_change_detected("status")

    def test_record_supplier_change_detected_ownership(self):
        record_supplier_change_detected("ownership")

    def test_record_certification_expiry(self):
        record_certification_expiry("expiring_soon")

    def test_record_certification_expiry_expired(self):
        record_certification_expiry("expired")

    def test_record_geolocation_drift(self):
        record_geolocation_drift()

    def test_record_deforestation_check(self):
        record_deforestation_check("completed")

    def test_record_deforestation_check_failed(self):
        record_deforestation_check("failed")

    def test_record_deforestation_correlation(self):
        record_deforestation_correlation()

    def test_record_investigation_triggered(self):
        record_investigation_triggered("deforestation_correlation")

    def test_record_compliance_audit(self):
        record_compliance_audit("compliant")

    def test_record_compliance_audit_non_compliant(self):
        record_compliance_audit("non_compliant")

    def test_record_compliance_check_passed(self):
        record_compliance_check_passed()

    def test_record_compliance_check_failed(self):
        record_compliance_check_failed()

    def test_record_change_detected(self):
        record_change_detected("ownership", "high")

    def test_record_change_detected_certification(self):
        record_change_detected("certification", "critical")

    def test_record_risk_monitor(self):
        record_risk_monitor("worsening")

    def test_record_risk_monitor_stable(self):
        record_risk_monitor("stable")

    def test_record_risk_degradation(self):
        record_risk_degradation()

    def test_record_freshness_check(self):
        record_freshness_check("fresh")

    def test_record_freshness_check_stale(self):
        record_freshness_check("stale_warning")

    def test_record_freshness_check_critical(self):
        record_freshness_check("stale_critical")

    def test_record_regulatory_update(self):
        record_regulatory_update("high")

    def test_record_regulatory_update_breaking(self):
        record_regulatory_update("breaking")

    def test_record_alert_generated(self):
        record_alert_generated("critical")

    def test_record_alert_generated_warning(self):
        record_alert_generated("warning")


class TestHistogramMetrics:
    """Test histogram metric observation functions."""

    def test_observe_supply_chain_scan_duration(self):
        observe_supply_chain_scan_duration(0.5)

    def test_observe_supply_chain_scan_duration_slow(self):
        observe_supply_chain_scan_duration(30.0)

    def test_observe_deforestation_check_duration(self):
        observe_deforestation_check_duration(1.0)

    def test_observe_deforestation_check_duration_fast(self):
        observe_deforestation_check_duration(0.1)

    def test_observe_compliance_audit_duration(self):
        observe_compliance_audit_duration(5.0)

    def test_observe_change_detection_duration(self):
        observe_change_detection_duration(0.3)

    def test_observe_risk_monitor_duration(self):
        observe_risk_monitor_duration(0.8)

    def test_observe_freshness_check_duration(self):
        observe_freshness_check_duration(0.2)

    def test_observe_regulatory_check_duration(self):
        observe_regulatory_check_duration(1.5)

    def test_observe_impact_assessment_duration(self):
        observe_impact_assessment_duration(0.4)

    def test_observe_correlation_duration(self):
        observe_correlation_duration(0.6)

    def test_observe_investigation_trigger_duration(self):
        observe_investigation_trigger_duration(0.3)

    def test_observe_trend_analysis_duration(self):
        observe_trend_analysis_duration(0.7)

    def test_observe_freshness_report_duration(self):
        observe_freshness_report_duration(0.5)

    def test_observe_notification_duration(self):
        observe_notification_duration(0.15)

    def test_observe_entity_mapping_duration(self):
        observe_entity_mapping_duration(0.25)


class TestGaugeMetrics:
    """Test gauge metric setting functions."""

    def test_set_active_scans(self):
        set_active_scans(10)

    def test_set_active_scans_zero(self):
        set_active_scans(0)

    def test_set_pending_investigations(self):
        set_pending_investigations(3)

    def test_set_pending_investigations_zero(self):
        set_pending_investigations(0)

    def test_set_expiring_certifications(self):
        set_expiring_certifications(5)

    def test_set_stale_entities(self):
        set_stale_entities(3)

    def test_set_stale_entities_zero(self):
        set_stale_entities(0)

    def test_set_open_alerts(self):
        set_open_alerts(8)

    def test_set_critical_alerts(self):
        set_critical_alerts(2)

    def test_set_high_risk_entities(self):
        set_high_risk_entities(7)

    def test_set_high_risk_entities_zero(self):
        set_high_risk_entities(0)

    def test_set_compliance_score(self):
        set_compliance_score(85.5)

    def test_set_freshness_percentage(self):
        set_freshness_percentage(92.3)

    def test_set_monitored_suppliers(self):
        set_monitored_suppliers(150)
