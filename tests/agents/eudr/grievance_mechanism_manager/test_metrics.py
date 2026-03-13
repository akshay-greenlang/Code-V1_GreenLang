# -*- coding: utf-8 -*-
"""
Unit tests for Prometheus metrics - AGENT-EUDR-032

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.grievance_mechanism_manager.metrics import (
    record_analytics_created,
    record_root_cause_analyzed,
    record_mediation_initiated,
    record_mediation_completed,
    record_remediation_created,
    record_remediation_verified,
    record_risk_score_computed,
    record_collective_created,
    record_regulatory_report_generated,
    observe_analytics_duration,
    observe_root_cause_duration,
    observe_mediation_session_duration,
    observe_remediation_verification_duration,
    observe_risk_scoring_duration,
    observe_report_generation_duration,
    set_active_mediations,
    set_open_remediations,
    set_high_risk_entities,
)


class TestCounterMetrics:
    def test_record_analytics_created(self):
        record_analytics_created("recurring")

    def test_record_root_cause_analyzed(self):
        record_root_cause_analyzed("five_whys")

    def test_record_mediation_initiated(self):
        record_mediation_initiated("internal")

    def test_record_mediation_completed(self):
        record_mediation_completed("accepted")

    def test_record_remediation_created(self):
        record_remediation_created("compensation")

    def test_record_remediation_verified(self):
        record_remediation_verified("verified")

    def test_record_risk_score_computed(self):
        record_risk_score_computed("operator")

    def test_record_collective_created(self):
        record_collective_created()

    def test_record_regulatory_report_generated(self):
        record_regulatory_report_generated("annual_summary")


class TestHistogramMetrics:
    def test_observe_analytics_duration(self):
        observe_analytics_duration(0.5)

    def test_observe_root_cause_duration(self):
        observe_root_cause_duration(1.0)

    def test_observe_mediation_session_duration(self):
        observe_mediation_session_duration(3600.0)

    def test_observe_remediation_verification_duration(self):
        observe_remediation_verification_duration(2.0)

    def test_observe_risk_scoring_duration(self):
        observe_risk_scoring_duration(0.3)

    def test_observe_report_generation_duration(self):
        observe_report_generation_duration("annual_summary", 5.0)


class TestGaugeMetrics:
    def test_set_active_mediations(self):
        set_active_mediations(10)

    def test_set_open_remediations(self):
        set_open_remediations(5)

    def test_set_high_risk_entities(self):
        set_high_risk_entities(3)
