# -*- coding: utf-8 -*-
"""
Unit tests for Engine 4: NonConformanceDetectionEngine -- AGENT-EUDR-024

Tests NC severity classification, EUDR article mapping, root cause analysis
frameworks, risk impact scoring, NC dispute handling, pattern detection,
and deterministic classification guarantee.

Target: ~60 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from decimal import Decimal

import pytest

from greenlang.agents.eudr.third_party_audit_manager.non_conformance_detection_engine import (
    NonConformanceDetectionEngine,
    CRITICAL_RULES,
    MAJOR_RULES,
)
from greenlang.agents.eudr.third_party_audit_manager.models import (
    ClassifyNCRequest,
    NCSeverity,
    NonConformance,
    RootCauseAnalysis,
)
from tests.agents.eudr.third_party_audit_manager.conftest import (
    SHA256_HEX_LENGTH,
    NC_SEVERITIES,
)


class TestNCEngineInit:
    """Test engine initialization."""

    def test_init_with_config(self, default_config):
        engine = NonConformanceDetectionEngine(config=default_config)
        assert engine.config is not None

    def test_init_without_config(self):
        engine = NonConformanceDetectionEngine()
        assert engine.config is not None

    def test_critical_rules_defined(self):
        assert len(CRITICAL_RULES) >= 7

    def test_major_rules_defined(self):
        assert len(MAJOR_RULES) >= 6

    def test_all_rules_have_rule_id(self):
        for rule in CRITICAL_RULES:
            assert "rule_id" in rule
            assert rule["rule_id"].startswith("CRIT-")
        for rule in MAJOR_RULES:
            assert "rule_id" in rule
            assert rule["rule_id"].startswith("MAJ-")


class TestCriticalClassification:
    """Test CRITICAL severity classification rules."""

    def test_fraud_classified_as_critical(self, nc_engine, classify_nc_fraud_request):
        response = nc_engine.classify(classify_nc_fraud_request)
        assert response.severity == NCSeverity.CRITICAL.value or response.severity == "critical"

    def test_deforestation_classified_as_critical(self, nc_engine, classify_nc_deforestation_request):
        response = nc_engine.classify(classify_nc_deforestation_request)
        assert response.severity in ("critical", NCSeverity.CRITICAL.value)

    def test_critical_nc_has_rule_id(self, nc_engine, classify_nc_fraud_request):
        response = nc_engine.classify(classify_nc_fraud_request)
        assert response.classification_rule is not None
        assert response.classification_rule.startswith("CRIT-")

    def test_critical_nc_maps_to_eudr_article(self, nc_engine, classify_nc_fraud_request):
        response = nc_engine.classify(classify_nc_fraud_request)
        assert response.eudr_article is not None

    def test_systematic_dds_failure_is_critical(self, nc_engine):
        request = ClassifyNCRequest(
            audit_id="AUD-TEST-001",
            finding_statement="Systematic failure of due diligence system",
            objective_evidence="No DDS documented for any supply chain",
            indicators={"systematic_dds_failure": True},
        )
        response = nc_engine.classify(request)
        assert response.severity in ("critical", NCSeverity.CRITICAL.value)

    def test_missing_all_geolocation_is_critical(self, nc_engine):
        request = ClassifyNCRequest(
            audit_id="AUD-TEST-001",
            finding_statement="Missing geolocation for all plots",
            objective_evidence="No GPS coordinates recorded for any production plot",
            indicators={"missing_all_geolocation": True},
        )
        response = nc_engine.classify(request)
        assert response.severity in ("critical", NCSeverity.CRITICAL.value)

    def test_authority_order_non_compliance_is_critical(self, nc_engine):
        request = ClassifyNCRequest(
            audit_id="AUD-TEST-001",
            finding_statement="Non-compliance with authority corrective action order",
            objective_evidence="BMEL corrective action order from 2025-11 not addressed",
            indicators={"authority_order_non_compliance": True},
        )
        response = nc_engine.classify(request)
        assert response.severity in ("critical", NCSeverity.CRITICAL.value)

    def test_certificate_falsification_is_critical(self, nc_engine):
        request = ClassifyNCRequest(
            audit_id="AUD-TEST-001",
            finding_statement="Falsified FSC certificate",
            objective_evidence="Certificate number not found in FSC database",
            indicators={"certificate_falsification": True},
        )
        response = nc_engine.classify(request)
        assert response.severity in ("critical", NCSeverity.CRITICAL.value)


class TestMajorClassification:
    """Test MAJOR severity classification rules."""

    def test_incomplete_risk_assessment_is_major(self, nc_engine, classify_nc_incomplete_risk_request):
        response = nc_engine.classify(classify_nc_incomplete_risk_request)
        assert response.severity in ("major", NCSeverity.MAJOR.value)

    def test_missing_supplier_info_is_major(self, nc_engine):
        request = ClassifyNCRequest(
            audit_id="AUD-TEST-001",
            finding_statement="Missing supplier information for Art. 9",
            objective_evidence="Supplier information incomplete for 3 suppliers",
            indicators={"missing_supplier_information": True},
        )
        response = nc_engine.classify(request)
        assert response.severity in ("major", NCSeverity.MAJOR.value)

    def test_expired_certification_is_major(self, nc_engine):
        request = ClassifyNCRequest(
            audit_id="AUD-TEST-001",
            finding_statement="Expired certification without renewal",
            objective_evidence="FSC certificate expired 6 months ago",
            indicators={"expired_certification_no_renewal": True},
        )
        response = nc_engine.classify(request)
        assert response.severity in ("major", NCSeverity.MAJOR.value)

    def test_major_nc_has_rule_id(self, nc_engine, classify_nc_incomplete_risk_request):
        response = nc_engine.classify(classify_nc_incomplete_risk_request)
        assert response.classification_rule is not None
        assert response.classification_rule.startswith("MAJ-")


class TestMinorClassification:
    """Test MINOR severity classification rules."""

    def test_training_records_is_minor(self, nc_engine, classify_nc_minor_request):
        response = nc_engine.classify(classify_nc_minor_request)
        assert response.severity in ("minor", NCSeverity.MINOR.value)

    def test_documentation_gap_is_minor(self, nc_engine):
        request = ClassifyNCRequest(
            audit_id="AUD-TEST-001",
            finding_statement="Minor documentation gaps in record keeping",
            objective_evidence="3 out of 50 transaction records incomplete",
            indicators={"isolated_documentation_gap": True},
        )
        response = nc_engine.classify(request)
        assert response.severity in ("minor", NCSeverity.MINOR.value, "observation")


class TestObservationClassification:
    """Test OBSERVATION classification for non-NC findings."""

    def test_no_indicators_returns_observation(self, nc_engine):
        request = ClassifyNCRequest(
            audit_id="AUD-TEST-001",
            finding_statement="Paper-based records could be digitized",
            objective_evidence="Manual record-keeping system observed",
            indicators={},
        )
        response = nc_engine.classify(request)
        assert response.severity in ("observation", NCSeverity.OBSERVATION.value)


class TestRiskImpactScore:
    """Test NC risk impact score calculation."""

    def test_critical_nc_high_impact(self, nc_engine, sample_nc_critical):
        score = nc_engine.calculate_risk_impact(
            severity="critical",
            eudr_article_criticality=Decimal("90"),
            supply_chain_volume_pct=Decimal("50"),
            supplier_risk_level=Decimal("80"),
        )
        assert score >= Decimal("60")

    def test_minor_nc_low_impact(self, nc_engine):
        score = nc_engine.calculate_risk_impact(
            severity="minor",
            eudr_article_criticality=Decimal("20"),
            supply_chain_volume_pct=Decimal("5"),
            supplier_risk_level=Decimal("30"),
        )
        assert score < Decimal("50")

    def test_impact_score_capped_at_100(self, nc_engine):
        score = nc_engine.calculate_risk_impact(
            severity="critical",
            eudr_article_criticality=Decimal("100"),
            supply_chain_volume_pct=Decimal("100"),
            supplier_risk_level=Decimal("100"),
        )
        assert score <= Decimal("100")

    def test_impact_score_is_decimal(self, nc_engine):
        score = nc_engine.calculate_risk_impact(
            severity="major",
            eudr_article_criticality=Decimal("50"),
            supply_chain_volume_pct=Decimal("30"),
            supplier_risk_level=Decimal("50"),
        )
        assert isinstance(score, Decimal)

    def test_observation_has_zero_severity_weight(self, nc_engine):
        score = nc_engine.calculate_risk_impact(
            severity="observation",
            eudr_article_criticality=Decimal("50"),
            supply_chain_volume_pct=Decimal("50"),
            supplier_risk_level=Decimal("50"),
        )
        # Observation has 0 severity weight, so impact is from other factors only
        assert score < Decimal("50")


class TestRootCauseAnalysis:
    """Test root cause analysis framework support."""

    def test_five_whys_framework(self, sample_rca_five_whys):
        assert sample_rca_five_whys.framework == "five_whys"
        assert len(sample_rca_five_whys.five_whys) == 5

    def test_ishikawa_framework(self, sample_rca_ishikawa):
        assert sample_rca_ishikawa.framework == "ishikawa"
        assert "people" in sample_rca_ishikawa.ishikawa_categories
        assert "process" in sample_rca_ishikawa.ishikawa_categories
        assert "equipment" in sample_rca_ishikawa.ishikawa_categories
        assert "materials" in sample_rca_ishikawa.ishikawa_categories
        assert "environment" in sample_rca_ishikawa.ishikawa_categories
        assert "management" in sample_rca_ishikawa.ishikawa_categories

    def test_rca_has_root_cause(self, sample_rca_five_whys):
        assert sample_rca_five_whys.root_cause is not None

    def test_rca_has_recommended_actions(self, sample_rca_five_whys):
        assert len(sample_rca_five_whys.recommended_actions) > 0

    def test_create_five_whys_rca(self, nc_engine):
        rca = nc_engine.create_root_cause_analysis(
            nc_id="NC-MAJ-001",
            framework="five_whys",
            analyst_id="AUR-FSC-001",
        )
        assert rca is not None
        assert rca.framework == "five_whys"

    def test_create_ishikawa_rca(self, nc_engine):
        rca = nc_engine.create_root_cause_analysis(
            nc_id="NC-CRIT-001",
            framework="ishikawa",
            analyst_id="AUR-FSC-001",
        )
        assert rca is not None
        assert rca.framework == "ishikawa"


class TestNCDispute:
    """Test NC dispute handling."""

    def test_nc_dispute_flag(self, nc_engine, sample_nc_major):
        result = nc_engine.dispute_nc(
            nc=sample_nc_major,
            dispute_rationale="Risk assessment was documented in a separate system not reviewed during audit",
        )
        assert result.disputed is True
        assert result.dispute_rationale is not None

    def test_nc_dispute_rationale_required(self, nc_engine, sample_nc_major):
        with pytest.raises((ValueError, Exception)):
            nc_engine.dispute_nc(nc=sample_nc_major, dispute_rationale="")


class TestNCProvenance:
    """Test NC provenance and determinism."""

    def test_classification_has_provenance(self, nc_engine, classify_nc_fraud_request):
        response = nc_engine.classify(classify_nc_fraud_request)
        assert response.provenance_hash is not None
        assert len(response.provenance_hash) == SHA256_HEX_LENGTH

    def test_classification_deterministic(self, nc_engine, classify_nc_fraud_request):
        r1 = nc_engine.classify(classify_nc_fraud_request)
        r2 = nc_engine.classify(classify_nc_fraud_request)
        assert r1.severity == r2.severity
        assert r1.classification_rule == r2.classification_rule


class TestNCBatchClassification:
    """Test batch NC classification."""

    def test_classify_batch(self, nc_engine):
        requests = [
            ClassifyNCRequest(
                audit_id="AUD-TEST-001",
                finding_statement=f"Finding {i}",
                objective_evidence=f"Evidence {i}",
                indicators={"fraud_or_falsification": True} if i == 0 else {"training_records_not_current": True},
            )
            for i in range(5)
        ]
        results = nc_engine.classify_batch(requests)
        assert len(results) == 5
        assert results[0].severity in ("critical", NCSeverity.CRITICAL.value)
