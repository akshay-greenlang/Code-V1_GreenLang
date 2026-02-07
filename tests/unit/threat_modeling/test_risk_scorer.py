"""
Unit tests for RiskScorer.

Tests risk score calculation, CVSS vector processing, likelihood and impact
assessment, and threat prioritization functionality.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

from greenlang.infrastructure.threat_modeling.risk_scorer import (
    RiskScorer,
    CVSSCalculator,
)
from greenlang.infrastructure.threat_modeling.models import (
    Threat,
    ThreatCategory,
    RiskLevel,
    CVSSVector,
    Component,
    ComponentType,
)


class TestRiskScorerInitialization:
    """Test RiskScorer initialization."""

    def test_initialization_with_config(self, risk_scorer_config):
        """Test scorer initializes with configuration."""
        scorer = RiskScorer(config=risk_scorer_config)

        assert scorer.config == risk_scorer_config
        assert scorer.likelihood_weights == risk_scorer_config.likelihood_weights
        assert scorer.impact_weights == risk_scorer_config.impact_weights

    def test_initialization_default_config(self):
        """Test scorer initializes with default configuration."""
        scorer = RiskScorer()

        assert scorer.config is not None
        assert scorer.likelihood_weights is not None
        assert scorer.impact_weights is not None

    def test_initialization_sets_risk_thresholds(self, risk_scorer_config):
        """Test scorer sets risk thresholds from config."""
        scorer = RiskScorer(config=risk_scorer_config)

        assert scorer.risk_thresholds == risk_scorer_config.risk_thresholds


class TestLikelihoodCalculation:
    """Test likelihood calculation."""

    def test_calculate_likelihood_high_complexity(self, risk_scorer_config):
        """Test likelihood for high complexity attack."""
        scorer = RiskScorer(config=risk_scorer_config)

        threat = Threat(
            threat_id=str(uuid4()),
            title="Complex Attack",
            description="Requires advanced skills",
            category=ThreatCategory.SPOOFING,
            affected_component_ids=[],
            affected_data_flow_ids=[],
            attack_vector="Requires physical access and admin credentials",
            prerequisites=["Physical access", "Admin credentials", "Specialized tools"],
            potential_impact="Data breach",
            likelihood=0.0,
            impact=0.0,
            risk_score=0.0,
            risk_level=RiskLevel.LOW,
            mitigations=[],
            status="identified",
            identified_at=datetime.utcnow(),
            identified_by="test",
        )

        likelihood = scorer.calculate_likelihood(threat)

        # High complexity should result in lower likelihood
        assert 0 <= likelihood <= 1
        assert likelihood < 0.5

    def test_calculate_likelihood_low_complexity(self, risk_scorer_config):
        """Test likelihood for low complexity attack."""
        scorer = RiskScorer(config=risk_scorer_config)

        threat = Threat(
            threat_id=str(uuid4()),
            title="Simple Attack",
            description="Easily exploitable",
            category=ThreatCategory.DENIAL_OF_SERVICE,
            affected_component_ids=[],
            affected_data_flow_ids=[],
            attack_vector="Network-based, no authentication required",
            prerequisites=[],  # No prerequisites
            potential_impact="Service unavailability",
            likelihood=0.0,
            impact=0.0,
            risk_score=0.0,
            risk_level=RiskLevel.LOW,
            mitigations=[],
            status="identified",
            identified_at=datetime.utcnow(),
            identified_by="test",
        )

        likelihood = scorer.calculate_likelihood(threat)

        # Low complexity should result in higher likelihood
        assert 0 <= likelihood <= 1
        assert likelihood > 0.3

    def test_calculate_likelihood_considers_prerequisites(self, risk_scorer_config):
        """Test likelihood considers number of prerequisites."""
        scorer = RiskScorer(config=risk_scorer_config)

        threat_few_prereqs = Threat(
            threat_id=str(uuid4()),
            title="Few Prerequisites",
            description="Easy to execute",
            category=ThreatCategory.SPOOFING,
            affected_component_ids=[],
            affected_data_flow_ids=[],
            attack_vector="Network",
            prerequisites=["Network access"],
            potential_impact="Access",
            likelihood=0.0,
            impact=0.0,
            risk_score=0.0,
            risk_level=RiskLevel.LOW,
            mitigations=[],
            status="identified",
            identified_at=datetime.utcnow(),
            identified_by="test",
        )

        threat_many_prereqs = Threat(
            threat_id=str(uuid4()),
            title="Many Prerequisites",
            description="Hard to execute",
            category=ThreatCategory.SPOOFING,
            affected_component_ids=[],
            affected_data_flow_ids=[],
            attack_vector="Complex",
            prerequisites=[
                "Physical access",
                "Admin credentials",
                "Internal knowledge",
                "Specialized tools",
                "Time window",
            ],
            potential_impact="Access",
            likelihood=0.0,
            impact=0.0,
            risk_score=0.0,
            risk_level=RiskLevel.LOW,
            mitigations=[],
            status="identified",
            identified_at=datetime.utcnow(),
            identified_by="test",
        )

        likelihood_few = scorer.calculate_likelihood(threat_few_prereqs)
        likelihood_many = scorer.calculate_likelihood(threat_many_prereqs)

        assert likelihood_few > likelihood_many

    def test_calculate_likelihood_returns_valid_range(self, risk_scorer_config):
        """Test likelihood is always in valid range."""
        scorer = RiskScorer(config=risk_scorer_config)

        for category in ThreatCategory:
            threat = Threat(
                threat_id=str(uuid4()),
                title="Test Threat",
                description="Test",
                category=category,
                affected_component_ids=[],
                affected_data_flow_ids=[],
                attack_vector="Test",
                prerequisites=[],
                potential_impact="Test",
                likelihood=0.0,
                impact=0.0,
                risk_score=0.0,
                risk_level=RiskLevel.LOW,
                mitigations=[],
                status="identified",
                identified_at=datetime.utcnow(),
                identified_by="test",
            )

            likelihood = scorer.calculate_likelihood(threat)

            assert 0 <= likelihood <= 1


class TestImpactCalculation:
    """Test impact calculation."""

    def test_calculate_impact_high_confidentiality(self, risk_scorer_config):
        """Test impact for high confidentiality threat."""
        scorer = RiskScorer(config=risk_scorer_config)

        threat = Threat(
            threat_id=str(uuid4()),
            title="Data Breach",
            description="Exposure of sensitive data",
            category=ThreatCategory.INFORMATION_DISCLOSURE,
            affected_component_ids=[],
            affected_data_flow_ids=[],
            attack_vector="Data exfiltration",
            prerequisites=[],
            potential_impact="Complete data breach of PII and financial records",
            likelihood=0.0,
            impact=0.0,
            risk_score=0.0,
            risk_level=RiskLevel.LOW,
            mitigations=[],
            status="identified",
            identified_at=datetime.utcnow(),
            identified_by="test",
        )

        impact = scorer.calculate_impact(threat)

        assert 0 <= impact <= 1
        assert impact > 0.5  # High impact

    def test_calculate_impact_considers_integrity(self, risk_scorer_config):
        """Test impact considers integrity effects."""
        scorer = RiskScorer(config=risk_scorer_config)

        threat = Threat(
            threat_id=str(uuid4()),
            title="Data Tampering",
            description="Modification of critical data",
            category=ThreatCategory.TAMPERING,
            affected_component_ids=[],
            affected_data_flow_ids=[],
            attack_vector="Man-in-the-middle",
            prerequisites=[],
            potential_impact="Financial record modification, audit trail corruption",
            likelihood=0.0,
            impact=0.0,
            risk_score=0.0,
            risk_level=RiskLevel.LOW,
            mitigations=[],
            status="identified",
            identified_at=datetime.utcnow(),
            identified_by="test",
        )

        impact = scorer.calculate_impact(threat)

        assert 0 <= impact <= 1

    def test_calculate_impact_considers_availability(self, risk_scorer_config):
        """Test impact considers availability effects."""
        scorer = RiskScorer(config=risk_scorer_config)

        threat = Threat(
            threat_id=str(uuid4()),
            title="Service Outage",
            description="Complete service unavailability",
            category=ThreatCategory.DENIAL_OF_SERVICE,
            affected_component_ids=[],
            affected_data_flow_ids=[],
            attack_vector="Resource exhaustion",
            prerequisites=[],
            potential_impact="24-hour service outage affecting all customers",
            likelihood=0.0,
            impact=0.0,
            risk_score=0.0,
            risk_level=RiskLevel.LOW,
            mitigations=[],
            status="identified",
            identified_at=datetime.utcnow(),
            identified_by="test",
        )

        impact = scorer.calculate_impact(threat)

        assert 0 <= impact <= 1
        assert impact > 0.3


class TestRiskScoreCalculation:
    """Test overall risk score calculation."""

    def test_calculate_risk_score_combines_likelihood_and_impact(
        self, risk_scorer_config, spoofing_threat
    ):
        """Test risk score combines likelihood and impact."""
        scorer = RiskScorer(config=risk_scorer_config)

        scored_threat = scorer.calculate_risk_score(spoofing_threat)

        assert scored_threat.risk_score >= 0
        assert scored_threat.risk_score <= 1
        # Risk should be related to likelihood * impact
        expected_approx = scored_threat.likelihood * scored_threat.impact
        assert abs(scored_threat.risk_score - expected_approx) < 0.3

    def test_calculate_risk_score_sets_likelihood(
        self, risk_scorer_config, tampering_threat
    ):
        """Test risk calculation sets likelihood."""
        scorer = RiskScorer(config=risk_scorer_config)

        original_likelihood = tampering_threat.likelihood
        scored_threat = scorer.calculate_risk_score(tampering_threat)

        assert scored_threat.likelihood != original_likelihood or original_likelihood > 0

    def test_calculate_risk_score_sets_impact(
        self, risk_scorer_config, dos_threat
    ):
        """Test risk calculation sets impact."""
        scorer = RiskScorer(config=risk_scorer_config)

        original_impact = dos_threat.impact
        scored_threat = scorer.calculate_risk_score(dos_threat)

        assert scored_threat.impact != original_impact or original_impact > 0

    def test_calculate_risk_score_sets_risk_level(
        self, risk_scorer_config, elevation_of_privilege_threat
    ):
        """Test risk calculation sets risk level."""
        scorer = RiskScorer(config=risk_scorer_config)

        scored_threat = scorer.calculate_risk_score(elevation_of_privilege_threat)

        assert scored_threat.risk_level is not None
        assert scored_threat.risk_level in [
            RiskLevel.CRITICAL,
            RiskLevel.HIGH,
            RiskLevel.MEDIUM,
            RiskLevel.LOW,
        ]

    @pytest.mark.parametrize("score,expected_level", [
        (0.95, RiskLevel.CRITICAL),
        (0.75, RiskLevel.HIGH),
        (0.5, RiskLevel.MEDIUM),
        (0.2, RiskLevel.LOW),
    ])
    def test_risk_score_to_level_mapping(
        self, risk_scorer_config, score, expected_level
    ):
        """Test risk scores map to correct levels."""
        scorer = RiskScorer(config=risk_scorer_config)

        level = scorer._score_to_level(score)

        assert level == expected_level


class TestCVSSCalculation:
    """Test CVSS score calculation."""

    def test_calculate_cvss_high_vector(self, high_cvss_vector):
        """Test CVSS calculation for high severity vector."""
        calculator = CVSSCalculator()

        score = calculator.calculate(high_cvss_vector)

        assert 7.0 <= score <= 10.0  # High/Critical range

    def test_calculate_cvss_medium_vector(self, medium_cvss_vector):
        """Test CVSS calculation for medium severity vector."""
        calculator = CVSSCalculator()

        score = calculator.calculate(medium_cvss_vector)

        assert 4.0 <= score <= 7.0  # Medium range

    def test_calculate_cvss_low_vector(self, low_cvss_vector):
        """Test CVSS calculation for low severity vector."""
        calculator = CVSSCalculator()

        score = calculator.calculate(low_cvss_vector)

        assert 0.0 <= score <= 4.0  # Low range

    def test_calculate_cvss_returns_valid_range(self):
        """Test CVSS score is always in valid range."""
        calculator = CVSSCalculator()

        # Test various combinations
        vectors = [
            CVSSVector(
                attack_vector=av,
                attack_complexity=ac,
                privileges_required=pr,
                user_interaction=ui,
                scope=s,
                confidentiality_impact=c,
                integrity_impact=i,
                availability_impact=a,
            )
            for av in ["NETWORK", "ADJACENT", "LOCAL", "PHYSICAL"]
            for ac in ["LOW", "HIGH"]
            for pr in ["NONE", "LOW", "HIGH"]
            for ui in ["NONE", "REQUIRED"]
            for s in ["UNCHANGED", "CHANGED"]
            for c in ["HIGH"]  # Fixed to reduce combinations
            for i in ["HIGH"]
            for a in ["HIGH"]
        ]

        for vector in vectors[:10]:  # Test subset
            score = calculator.calculate(vector)
            assert 0.0 <= score <= 10.0

    def test_calculate_cvss_attack_vector_impact(self):
        """Test attack vector affects CVSS score."""
        calculator = CVSSCalculator()

        network_vector = CVSSVector(
            attack_vector="NETWORK",
            attack_complexity="LOW",
            privileges_required="NONE",
            user_interaction="NONE",
            scope="UNCHANGED",
            confidentiality_impact="HIGH",
            integrity_impact="HIGH",
            availability_impact="HIGH",
        )

        physical_vector = CVSSVector(
            attack_vector="PHYSICAL",
            attack_complexity="LOW",
            privileges_required="NONE",
            user_interaction="NONE",
            scope="UNCHANGED",
            confidentiality_impact="HIGH",
            integrity_impact="HIGH",
            availability_impact="HIGH",
        )

        network_score = calculator.calculate(network_vector)
        physical_score = calculator.calculate(physical_vector)

        # Network attack vector should have higher score
        assert network_score > physical_score

    def test_calculate_cvss_with_threat(self, risk_scorer_config, spoofing_threat, high_cvss_vector):
        """Test CVSS calculation integrated with risk scorer."""
        scorer = RiskScorer(config=risk_scorer_config)

        cvss_score = scorer.calculate_cvss(spoofing_threat, high_cvss_vector)

        assert 0.0 <= cvss_score <= 10.0


class TestThreatPrioritization:
    """Test threat prioritization functionality."""

    def test_prioritize_threats_orders_by_risk(
        self, risk_scorer_config, all_stride_threats
    ):
        """Test threats are prioritized by risk score."""
        scorer = RiskScorer(config=risk_scorer_config)

        prioritized = scorer.prioritize_threats(all_stride_threats)

        # Should be in descending order of risk
        for i in range(len(prioritized) - 1):
            assert prioritized[i].risk_score >= prioritized[i + 1].risk_score

    def test_prioritize_threats_calculates_scores(
        self, risk_scorer_config, spoofing_threat, tampering_threat
    ):
        """Test prioritization calculates risk scores."""
        scorer = RiskScorer(config=risk_scorer_config)

        threats = [spoofing_threat, tampering_threat]
        prioritized = scorer.prioritize_threats(threats)

        for threat in prioritized:
            assert threat.risk_score > 0

    def test_prioritize_empty_list(self, risk_scorer_config):
        """Test prioritizing empty list."""
        scorer = RiskScorer(config=risk_scorer_config)

        prioritized = scorer.prioritize_threats([])

        assert prioritized == []

    def test_prioritize_single_threat(self, risk_scorer_config, spoofing_threat):
        """Test prioritizing single threat."""
        scorer = RiskScorer(config=risk_scorer_config)

        prioritized = scorer.prioritize_threats([spoofing_threat])

        assert len(prioritized) == 1
        assert prioritized[0].risk_score > 0

    def test_prioritize_considers_business_context(self, risk_scorer_config):
        """Test prioritization considers business context."""
        scorer = RiskScorer(config=risk_scorer_config)

        prod_threat = Threat(
            threat_id=str(uuid4()),
            title="Production Threat",
            description="Threat in production",
            category=ThreatCategory.DENIAL_OF_SERVICE,
            affected_component_ids=[],
            affected_data_flow_ids=[],
            attack_vector="Network",
            prerequisites=[],
            potential_impact="Service outage",
            likelihood=0.5,
            impact=0.5,
            risk_score=0.0,
            risk_level=RiskLevel.MEDIUM,
            mitigations=[],
            status="identified",
            identified_at=datetime.utcnow(),
            identified_by="test",
        )

        dev_threat = Threat(
            threat_id=str(uuid4()),
            title="Development Threat",
            description="Threat in development",
            category=ThreatCategory.DENIAL_OF_SERVICE,
            affected_component_ids=[],
            affected_data_flow_ids=[],
            attack_vector="Network",
            prerequisites=[],
            potential_impact="Service outage",
            likelihood=0.5,
            impact=0.5,
            risk_score=0.0,
            risk_level=RiskLevel.MEDIUM,
            mitigations=[],
            status="identified",
            identified_at=datetime.utcnow(),
            identified_by="test",
        )

        # Add business context
        prod_threat_with_context = scorer.apply_business_context(
            prod_threat, environment="production"
        )
        dev_threat_with_context = scorer.apply_business_context(
            dev_threat, environment="development"
        )

        # Production should have higher adjusted score
        assert prod_threat_with_context.risk_score >= dev_threat_with_context.risk_score


class TestBusinessContextMultipliers:
    """Test business context multiplier application."""

    def test_production_multiplier(self, risk_scorer_config, spoofing_threat):
        """Test production environment multiplier."""
        scorer = RiskScorer(config=risk_scorer_config)

        original = scorer.calculate_risk_score(spoofing_threat)
        adjusted = scorer.apply_business_context(original, environment="production")

        # Production multiplier should increase score
        assert adjusted.risk_score >= original.risk_score

    def test_development_multiplier(self, risk_scorer_config, spoofing_threat):
        """Test development environment multiplier."""
        scorer = RiskScorer(config=risk_scorer_config)

        original = scorer.calculate_risk_score(spoofing_threat)
        adjusted = scorer.apply_business_context(original, environment="development")

        # Development multiplier should decrease score
        assert adjusted.risk_score <= original.risk_score

    def test_staging_multiplier_neutral(self, risk_scorer_config, spoofing_threat):
        """Test staging environment multiplier is neutral."""
        scorer = RiskScorer(config=risk_scorer_config)

        original = scorer.calculate_risk_score(spoofing_threat)
        adjusted = scorer.apply_business_context(original, environment="staging")

        # Staging multiplier should be neutral (1.0)
        assert abs(adjusted.risk_score - original.risk_score) < 0.1


class TestMitigationEffectiveness:
    """Test mitigation effectiveness on risk scores."""

    def test_mitigation_reduces_risk_score(
        self, risk_scorer_config, spoofing_threat, sample_mitigation
    ):
        """Test implemented mitigation reduces risk score."""
        scorer = RiskScorer(config=risk_scorer_config)

        # Score without mitigation
        unmitigated = scorer.calculate_risk_score(spoofing_threat)

        # Add mitigation
        spoofing_threat.mitigations = [sample_mitigation]
        sample_mitigation.status = "implemented"
        mitigated = scorer.calculate_risk_score(spoofing_threat)

        # Mitigated score should be lower
        assert mitigated.risk_score <= unmitigated.risk_score

    def test_planned_mitigation_partial_effect(
        self, risk_scorer_config, spoofing_threat, sample_mitigation
    ):
        """Test planned mitigation has partial effect."""
        scorer = RiskScorer(config=risk_scorer_config)

        # Score without mitigation
        unmitigated = scorer.calculate_risk_score(spoofing_threat)

        # Add planned mitigation
        spoofing_threat.mitigations = [sample_mitigation]
        sample_mitigation.status = "planned"
        mitigated = scorer.calculate_risk_score(spoofing_threat)

        # Planned mitigation may have small effect
        assert mitigated.risk_score <= unmitigated.risk_score

    def test_multiple_mitigations_stack(self, risk_scorer_config, spoofing_threat):
        """Test multiple mitigations stack effectiveness."""
        scorer = RiskScorer(config=risk_scorer_config)

        from greenlang.infrastructure.threat_modeling.models import Mitigation, MitigationStatus

        mitigation1 = Mitigation(
            mitigation_id=str(uuid4()),
            title="Mitigation 1",
            description="First mitigation",
            threat_ids=[],
            control_type="preventive",
            implementation_effort="low",
            effectiveness=0.5,
            status=MitigationStatus.IMPLEMENTED,
            owner="team",
            due_date=None,
            metadata={},
        )

        mitigation2 = Mitigation(
            mitigation_id=str(uuid4()),
            title="Mitigation 2",
            description="Second mitigation",
            threat_ids=[],
            control_type="detective",
            implementation_effort="medium",
            effectiveness=0.3,
            status=MitigationStatus.IMPLEMENTED,
            owner="team",
            due_date=None,
            metadata={},
        )

        # Score with no mitigations
        unmitigated = scorer.calculate_risk_score(spoofing_threat)

        # Score with one mitigation
        spoofing_threat.mitigations = [mitigation1]
        one_mitigation = scorer.calculate_risk_score(spoofing_threat)

        # Score with two mitigations
        spoofing_threat.mitigations = [mitigation1, mitigation2]
        two_mitigations = scorer.calculate_risk_score(spoofing_threat)

        # More mitigations should reduce risk more
        assert two_mitigations.risk_score <= one_mitigation.risk_score <= unmitigated.risk_score


class TestRiskScorerEdgeCases:
    """Test edge cases for risk scorer."""

    def test_handles_zero_likelihood(self, risk_scorer_config):
        """Test handling threat with zero likelihood."""
        scorer = RiskScorer(config=risk_scorer_config)

        threat = Threat(
            threat_id=str(uuid4()),
            title="Impossible Threat",
            description="Cannot happen",
            category=ThreatCategory.SPOOFING,
            affected_component_ids=[],
            affected_data_flow_ids=[],
            attack_vector="None",
            prerequisites=["Impossible conditions"],
            potential_impact="None",
            likelihood=0.0,
            impact=1.0,
            risk_score=0.0,
            risk_level=RiskLevel.LOW,
            mitigations=[],
            status="identified",
            identified_at=datetime.utcnow(),
            identified_by="test",
        )

        scored = scorer.calculate_risk_score(threat)

        assert scored.risk_score >= 0

    def test_handles_zero_impact(self, risk_scorer_config):
        """Test handling threat with zero impact."""
        scorer = RiskScorer(config=risk_scorer_config)

        threat = Threat(
            threat_id=str(uuid4()),
            title="No Impact Threat",
            description="No effect",
            category=ThreatCategory.DENIAL_OF_SERVICE,
            affected_component_ids=[],
            affected_data_flow_ids=[],
            attack_vector="Simple",
            prerequisites=[],
            potential_impact="No effect",
            likelihood=1.0,
            impact=0.0,
            risk_score=0.0,
            risk_level=RiskLevel.LOW,
            mitigations=[],
            status="identified",
            identified_at=datetime.utcnow(),
            identified_by="test",
        )

        scored = scorer.calculate_risk_score(threat)

        assert scored.risk_score >= 0

    def test_handles_extreme_values(self, risk_scorer_config):
        """Test handling extreme likelihood and impact values."""
        scorer = RiskScorer(config=risk_scorer_config)

        threat = Threat(
            threat_id=str(uuid4()),
            title="Extreme Threat",
            description="Maximum severity",
            category=ThreatCategory.ELEVATION_OF_PRIVILEGE,
            affected_component_ids=[],
            affected_data_flow_ids=[],
            attack_vector="Trivial",
            prerequisites=[],
            potential_impact="Complete system compromise",
            likelihood=1.0,
            impact=1.0,
            risk_score=0.0,
            risk_level=RiskLevel.LOW,
            mitigations=[],
            status="identified",
            identified_at=datetime.utcnow(),
            identified_by="test",
        )

        scored = scorer.calculate_risk_score(threat)

        assert scored.risk_level == RiskLevel.CRITICAL
        assert scored.risk_score >= 0.9
