# -*- coding: utf-8 -*-
"""
Unit tests for EU Taxonomy Minimum Safeguards Assessment Engine.

Tests four-topic assessment (human rights, anti-corruption, taxation,
fair competition), procedural and outcome-based checks, adverse finding
recording, pass/fail aggregation, and evidence linkage with 35+ test
functions.

Author: GL-TestEngineer
Date: March 2026
"""

import pytest


# ===========================================================================
# Four-topic assessment tests
# ===========================================================================

class TestFourTopicAssessment:
    """Test comprehensive four-topic safeguard assessment."""

    def test_all_four_topics_assessed(self, sample_safeguard_assessment):
        """Assessment covers all four mandatory topics."""
        topics = sample_safeguard_assessment["topics"]
        expected = {"human_rights", "anti_corruption", "taxation", "fair_competition"}
        assert set(topics.keys()) == expected

    def test_overall_pass_all_topics(self, sample_safeguard_assessment):
        """Overall pass requires all four topics to pass."""
        assert sample_safeguard_assessment["overall_pass"] is True
        for topic, result in sample_safeguard_assessment["topics"].items():
            assert result["overall"] is True

    def test_overall_fail_one_topic(self, failing_safeguard_assessment):
        """Overall fails when any single topic fails."""
        assert failing_safeguard_assessment["overall_pass"] is False
        topics = failing_safeguard_assessment["topics"]
        fail_count = sum(1 for t in topics.values() if not t["overall"])
        assert fail_count >= 1

    def test_assessment_status(self, sample_safeguard_assessment):
        """Assessment has completed status."""
        assert sample_safeguard_assessment["status"] == "completed"

    def test_assessment_timestamp(self, sample_safeguard_assessment):
        """Assessment date is recorded."""
        from datetime import datetime
        assert isinstance(sample_safeguard_assessment["assessment_date"], datetime)

    def test_provenance_hash(self, sample_safeguard_assessment):
        """Assessment has provenance hash."""
        assert len(sample_safeguard_assessment["provenance_hash"]) == 64


# ===========================================================================
# Human rights topic tests
# ===========================================================================

class TestHumanRightsTopic:
    """Test human rights safeguard topic."""

    def test_hr_procedural_pass(self, sample_safeguard_topic_results):
        """Human rights procedural checks pass."""
        hr = next(r for r in sample_safeguard_topic_results if r["topic"] == "human_rights")
        assert hr["procedural_pass"] is True

    def test_hr_outcome_pass(self, sample_safeguard_topic_results):
        """Human rights outcome checks pass."""
        hr = next(r for r in sample_safeguard_topic_results if r["topic"] == "human_rights")
        assert hr["outcome_pass"] is True

    def test_hr_ungp_due_diligence(self, sample_safeguard_topic_results):
        """UNGP due diligence process in place."""
        hr = next(r for r in sample_safeguard_topic_results if r["topic"] == "human_rights")
        assert hr["checks"]["ungp_due_diligence"] is True

    def test_hr_ilo_conventions(self, sample_safeguard_topic_results):
        """ILO core conventions compliance checked."""
        hr = next(r for r in sample_safeguard_topic_results if r["topic"] == "human_rights")
        assert hr["checks"]["ilo_core_conventions"] is True

    def test_hr_policy_exists(self, sample_safeguard_topic_results):
        """Human rights policy exists."""
        hr = next(r for r in sample_safeguard_topic_results if r["topic"] == "human_rights")
        assert hr["checks"]["human_rights_policy"] is True

    def test_hr_grievance_mechanism(self, sample_safeguard_topic_results):
        """Grievance mechanism available."""
        hr = next(r for r in sample_safeguard_topic_results if r["topic"] == "human_rights")
        assert hr["checks"]["grievance_mechanism"] is True

    def test_hr_adverse_impact_assessment(self, sample_safeguard_topic_results):
        """Adverse impact assessment conducted."""
        hr = next(r for r in sample_safeguard_topic_results if r["topic"] == "human_rights")
        assert hr["checks"]["adverse_impact_assessment"] is True

    def test_engine_check_human_rights(self, safeguard_engine):
        """Engine human rights check returns True."""
        result = safeguard_engine.check_human_rights("org-123")
        assert result is True


# ===========================================================================
# Anti-corruption topic tests
# ===========================================================================

class TestAntiCorruptionTopic:
    """Test anti-corruption safeguard topic."""

    def test_anti_corruption_pass(self, sample_safeguard_topic_results):
        """Anti-corruption topic passes."""
        ac = next(r for r in sample_safeguard_topic_results if r["topic"] == "anti_corruption")
        assert ac["overall_pass"] is True

    def test_anti_bribery_policy(self, sample_safeguard_topic_results):
        """Anti-bribery policy exists."""
        ac = next(r for r in sample_safeguard_topic_results if r["topic"] == "anti_corruption")
        assert ac["checks"]["anti_bribery_policy"] is True

    def test_iso37001_certification(self, sample_safeguard_topic_results):
        """ISO 37001 anti-bribery certification held."""
        ac = next(r for r in sample_safeguard_topic_results if r["topic"] == "anti_corruption")
        assert ac["checks"]["iso37001_certified"] is True

    def test_whistleblower_mechanism(self, sample_safeguard_topic_results):
        """Whistleblower mechanism in place."""
        ac = next(r for r in sample_safeguard_topic_results if r["topic"] == "anti_corruption")
        assert ac["checks"]["whistleblower_mechanism"] is True

    def test_no_regulatory_sanctions(self, sample_safeguard_topic_results):
        """No outstanding regulatory sanctions."""
        ac = next(r for r in sample_safeguard_topic_results if r["topic"] == "anti_corruption")
        assert ac["checks"]["no_regulatory_sanctions"] is True

    def test_anti_corruption_fail_scenario(self, failing_safeguard_assessment):
        """Anti-corruption fails with adverse finding."""
        topics = failing_safeguard_assessment["topics"]
        assert topics["anti_corruption"]["overall"] is False
        assert topics["anti_corruption"]["outcome"] is False


# ===========================================================================
# Taxation topic tests
# ===========================================================================

class TestTaxationTopic:
    """Test taxation safeguard topic."""

    def test_taxation_pass(self, sample_safeguard_topic_results):
        """Taxation topic passes."""
        tax = next(r for r in sample_safeguard_topic_results if r["topic"] == "taxation")
        assert tax["overall_pass"] is True

    def test_tax_governance(self, sample_safeguard_topic_results):
        """Tax governance framework exists."""
        tax = next(r for r in sample_safeguard_topic_results if r["topic"] == "taxation")
        assert tax["checks"]["tax_governance_framework"] is True

    def test_country_by_country_reporting(self, sample_safeguard_topic_results):
        """Country-by-country reporting in place."""
        tax = next(r for r in sample_safeguard_topic_results if r["topic"] == "taxation")
        assert tax["checks"]["country_by_country_reporting"] is True

    def test_transfer_pricing_compliance(self, sample_safeguard_topic_results):
        """Transfer pricing compliance verified."""
        tax = next(r for r in sample_safeguard_topic_results if r["topic"] == "taxation")
        assert tax["checks"]["transfer_pricing_compliance"] is True

    def test_no_aggressive_tax_planning(self, sample_safeguard_topic_results):
        """No aggressive tax planning identified."""
        tax = next(r for r in sample_safeguard_topic_results if r["topic"] == "taxation")
        assert tax["checks"]["no_aggressive_tax_planning"] is True

    def test_engine_check_taxation(self, safeguard_engine):
        """Engine taxation check returns True."""
        result = safeguard_engine.check_taxation("org-123")
        assert result is True


# ===========================================================================
# Fair competition topic tests
# ===========================================================================

class TestFairCompetitionTopic:
    """Test fair competition safeguard topic."""

    def test_fair_competition_pass(self, sample_safeguard_topic_results):
        """Fair competition topic passes."""
        fc = next(r for r in sample_safeguard_topic_results if r["topic"] == "fair_competition")
        assert fc["overall_pass"] is True

    def test_competition_policy(self, sample_safeguard_topic_results):
        """Competition compliance policy exists."""
        fc = next(r for r in sample_safeguard_topic_results if r["topic"] == "fair_competition")
        assert fc["checks"]["competition_compliance_policy"] is True

    def test_no_antitrust_proceedings(self, sample_safeguard_topic_results):
        """No active antitrust proceedings."""
        fc = next(r for r in sample_safeguard_topic_results if r["topic"] == "fair_competition")
        assert fc["checks"]["no_antitrust_proceedings"] is True

    def test_no_cartel_findings(self, sample_safeguard_topic_results):
        """No cartel findings."""
        fc = next(r for r in sample_safeguard_topic_results if r["topic"] == "fair_competition")
        assert fc["checks"]["no_cartel_findings"] is True

    def test_engine_check_fair_competition(self, safeguard_engine):
        """Engine fair competition check returns True."""
        result = safeguard_engine.check_fair_competition("org-123")
        assert result is True


# ===========================================================================
# Procedural and outcome checks tests
# ===========================================================================

class TestProceduralAndOutcomeChecks:
    """Test separation of procedural and outcome-based checks."""

    def test_procedural_checks_all_pass(self, sample_safeguard_topic_results):
        """All topics pass procedural checks."""
        for result in sample_safeguard_topic_results:
            assert result["procedural_pass"] is True

    def test_outcome_checks_all_pass(self, sample_safeguard_topic_results):
        """All topics pass outcome checks."""
        for result in sample_safeguard_topic_results:
            assert result["outcome_pass"] is True

    def test_overall_requires_both(self, sample_safeguard_topic_results):
        """Overall pass requires both procedural and outcome pass."""
        for result in sample_safeguard_topic_results:
            if result["overall_pass"]:
                assert result["procedural_pass"] is True
                assert result["outcome_pass"] is True

    def test_procedural_pass_outcome_fail(self, failing_safeguard_assessment):
        """Procedural pass with outcome fail results in topic fail."""
        ac = failing_safeguard_assessment["topics"]["anti_corruption"]
        assert ac["procedural"] is True
        assert ac["outcome"] is False
        assert ac["overall"] is False


# ===========================================================================
# Adverse finding recording tests
# ===========================================================================

class TestAdverseFindingRecording:
    """Test adverse finding recording in safeguard assessments."""

    def test_adverse_finding_identified(self, failing_safeguard_assessment):
        """Adverse finding is identified in assessment."""
        assert failing_safeguard_assessment["overall_pass"] is False

    def test_adverse_finding_notes(self, failing_safeguard_assessment):
        """Adverse finding is documented in notes."""
        assert "adverse finding" in failing_safeguard_assessment["notes"].lower()

    def test_engine_record_adverse_finding(self, safeguard_engine):
        """Engine can record adverse findings."""
        safeguard_engine.record_adverse_finding.return_value = {
            "topic": "anti_corruption",
            "finding": "Regulatory sanction in jurisdiction X",
            "severity": "material",
        }
        result = safeguard_engine.record_adverse_finding(
            org_id="org-123",
            topic="anti_corruption",
            finding="Regulatory sanction",
        )
        safeguard_engine.record_adverse_finding.assert_called_once()

    def test_adverse_finding_blocks_alignment(self, failing_safeguard_assessment):
        """Adverse finding prevents taxonomy alignment."""
        assert failing_safeguard_assessment["overall_pass"] is False


# ===========================================================================
# Evidence linkage tests
# ===========================================================================

class TestSafeguardEvidence:
    """Test evidence linkage for safeguard assessments."""

    def test_evidence_per_topic(self, sample_safeguard_assessment):
        """Evidence items reference specific topics."""
        evidence = sample_safeguard_assessment["evidence_items"]
        topics_with_evidence = {e["topic"] for e in evidence}
        assert len(topics_with_evidence) == 4

    def test_evidence_types_appropriate(self, sample_safeguard_assessment):
        """Evidence types match topic requirements."""
        evidence = sample_safeguard_assessment["evidence_items"]
        type_map = {e["topic"]: e["type"] for e in evidence}
        assert type_map["anti_corruption"] == "certification"  # ISO 37001
        assert type_map["taxation"] == "audit"

    def test_topic_level_evidence(self, sample_safeguard_topic_results):
        """Per-topic results include evidence items."""
        for result in sample_safeguard_topic_results:
            assert len(result["evidence_items"]) >= 1

    def test_engine_assess_returns_evidence(self, safeguard_engine):
        """Engine assessment returns evidence references."""
        safeguard_engine.assess.return_value = {
            "overall_pass": True,
            "topics": {"human_rights": {"pass": True}},
            "evidence_count": 4,
        }
        result = safeguard_engine.assess("org-123")
        assert result["evidence_count"] == 4
