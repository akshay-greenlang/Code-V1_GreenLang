"""
Unit tests for IncidentClassifier.

Tests incident type classification, severity assignment, escalation level
determination, and business impact calculation.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from uuid import uuid4

from greenlang.infrastructure.incident_response.classifier import (
    IncidentClassifier,
    SeverityLevel,
)
from greenlang.infrastructure.incident_response.models import (
    Alert,
    Incident,
    AlertSource,
    IncidentType,
    IncidentStatus,
    EscalationLevel,
)


class TestClassifierInitialization:
    """Test IncidentClassifier initialization."""

    def test_initialization_with_config(self, classifier_config):
        """Test classifier initializes with configuration."""
        classifier = IncidentClassifier(config=classifier_config)

        assert classifier.config == classifier_config
        assert classifier.severity_weights == classifier_config.severity_weights

    def test_initialization_default_config(self):
        """Test classifier initializes with default configuration."""
        classifier = IncidentClassifier()

        assert classifier.config is not None
        assert classifier.severity_weights is not None

    def test_initialization_loads_severity_levels(self, classifier_config):
        """Test classifier loads severity level definitions."""
        classifier = IncidentClassifier(config=classifier_config)

        assert hasattr(classifier, "SEVERITY_LEVELS")
        assert len(classifier.SEVERITY_LEVELS) > 0


class TestIncidentClassification:
    """Test incident type classification."""

    def test_classify_security_incident(self, classifier_config, security_alert):
        """Test classifying security-related incident."""
        classifier = IncidentClassifier(config=classifier_config)

        incident = Incident(
            incident_id=str(uuid4()),
            title="Security incident",
            description="Security alerts detected",
            incident_type=IncidentType.UNKNOWN,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P3,
            alerts=[security_alert],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=[],
            metadata={},
        )

        classified = classifier.classify(incident)

        assert classified.incident_type == IncidentType.SECURITY

    def test_classify_infrastructure_incident(self, classifier_config):
        """Test classifying infrastructure-related incident."""
        classifier = IncidentClassifier(config=classifier_config)

        infra_alert = Alert(
            alert_id=str(uuid4()),
            title="High CPU usage",
            description="CPU exceeded threshold",
            severity="high",
            source=AlertSource.PROMETHEUS,
            timestamp=datetime.utcnow(),
            labels={"alertname": "HighCPU", "instance": "node-1"},
            annotations={},
            raw_data={},
        )

        incident = Incident(
            incident_id=str(uuid4()),
            title="Infrastructure incident",
            description="Resource alerts detected",
            incident_type=IncidentType.UNKNOWN,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P3,
            alerts=[infra_alert],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=[],
            metadata={},
        )

        classified = classifier.classify(incident)

        assert classified.incident_type == IncidentType.INFRASTRUCTURE

    def test_classify_application_incident(self, classifier_config):
        """Test classifying application-related incident."""
        classifier = IncidentClassifier(config=classifier_config)

        app_alert = Alert(
            alert_id=str(uuid4()),
            title="Application error rate high",
            description="Error logs increased",
            severity="warning",
            source=AlertSource.LOKI,
            timestamp=datetime.utcnow(),
            labels={"app": "greenlang-api", "level": "error"},
            annotations={},
            raw_data={},
        )

        incident = Incident(
            incident_id=str(uuid4()),
            title="Application incident",
            description="Application errors detected",
            incident_type=IncidentType.UNKNOWN,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P3,
            alerts=[app_alert],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=[],
            metadata={},
        )

        classified = classifier.classify(incident)

        assert classified.incident_type == IncidentType.APPLICATION

    def test_classify_mixed_incident_prioritizes_security(self, classifier_config, security_alert):
        """Test mixed incident prioritizes security classification."""
        classifier = IncidentClassifier(config=classifier_config)

        infra_alert = Alert(
            alert_id=str(uuid4()),
            title="High CPU usage",
            description="CPU exceeded threshold",
            severity="warning",
            source=AlertSource.PROMETHEUS,
            timestamp=datetime.utcnow(),
            labels={},
            annotations={},
            raw_data={},
        )

        incident = Incident(
            incident_id=str(uuid4()),
            title="Mixed incident",
            description="Multiple alert types",
            incident_type=IncidentType.UNKNOWN,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P3,
            alerts=[security_alert, infra_alert],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=[],
            metadata={},
        )

        classified = classifier.classify(incident)

        # Security should take priority
        assert classified.incident_type == IncidentType.SECURITY


class TestEscalationLevelDetermination:
    """Test escalation level determination."""

    def test_critical_alerts_get_p0_escalation(self, classifier_config):
        """Test critical alerts result in P0 escalation."""
        classifier = IncidentClassifier(config=classifier_config)

        critical_alert = Alert(
            alert_id=str(uuid4()),
            title="Critical alert",
            description="Critical issue",
            severity="critical",
            source=AlertSource.GUARDDUTY,
            timestamp=datetime.utcnow(),
            labels={},
            annotations={},
            raw_data={},
        )

        incident = Incident(
            incident_id=str(uuid4()),
            title="Critical incident",
            description="Critical",
            incident_type=IncidentType.SECURITY,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P3,
            alerts=[critical_alert],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=[],
            metadata={},
        )

        classified = classifier.classify(incident)

        assert classified.escalation_level == EscalationLevel.P0

    def test_high_severity_gets_p1_escalation(self, classifier_config):
        """Test high severity alerts result in P1 escalation."""
        classifier = IncidentClassifier(config=classifier_config)

        high_alert = Alert(
            alert_id=str(uuid4()),
            title="High severity alert",
            description="High severity issue",
            severity="high",
            source=AlertSource.PROMETHEUS,
            timestamp=datetime.utcnow(),
            labels={},
            annotations={},
            raw_data={},
        )

        incident = Incident(
            incident_id=str(uuid4()),
            title="High severity incident",
            description="High",
            incident_type=IncidentType.INFRASTRUCTURE,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P3,
            alerts=[high_alert],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=[],
            metadata={},
        )

        classified = classifier.classify(incident)

        assert classified.escalation_level in [EscalationLevel.P0, EscalationLevel.P1]

    def test_warning_alerts_get_p2_escalation(self, classifier_config):
        """Test warning alerts result in P2 escalation."""
        classifier = IncidentClassifier(config=classifier_config)

        warning_alert = Alert(
            alert_id=str(uuid4()),
            title="Warning alert",
            description="Warning issue",
            severity="warning",
            source=AlertSource.PROMETHEUS,
            timestamp=datetime.utcnow(),
            labels={},
            annotations={},
            raw_data={},
        )

        incident = Incident(
            incident_id=str(uuid4()),
            title="Warning incident",
            description="Warning",
            incident_type=IncidentType.INFRASTRUCTURE,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P3,
            alerts=[warning_alert],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=[],
            metadata={},
        )

        classified = classifier.classify(incident)

        assert classified.escalation_level in [EscalationLevel.P2, EscalationLevel.P3]

    def test_multiple_alerts_escalate_higher(self, classifier_config):
        """Test multiple alerts can escalate to higher level."""
        classifier = IncidentClassifier(config=classifier_config)

        warning_alerts = [
            Alert(
                alert_id=str(uuid4()),
                title=f"Warning alert {i}",
                description="Warning issue",
                severity="warning",
                source=AlertSource.PROMETHEUS,
                timestamp=datetime.utcnow(),
                labels={"instance": f"node-{i}"},
                annotations={},
                raw_data={},
            )
            for i in range(5)
        ]

        incident = Incident(
            incident_id=str(uuid4()),
            title="Multiple warnings incident",
            description="Multiple warnings",
            incident_type=IncidentType.INFRASTRUCTURE,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P3,
            alerts=warning_alerts,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=[],
            metadata={},
        )

        classified = classifier.classify(incident)

        # Multiple warnings should escalate above P3
        assert classified.escalation_level.value <= EscalationLevel.P2.value

    def test_security_incidents_escalate_higher(self, classifier_config, security_alert):
        """Test security incidents get higher escalation."""
        classifier = IncidentClassifier(config=classifier_config)

        incident = Incident(
            incident_id=str(uuid4()),
            title="Security incident",
            description="Security issue",
            incident_type=IncidentType.SECURITY,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P3,
            alerts=[security_alert],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=[],
            metadata={},
        )

        classified = classifier.classify(incident)

        # Security incidents should be P1 or higher
        assert classified.escalation_level.value <= EscalationLevel.P1.value


class TestBusinessImpactCalculation:
    """Test business impact calculation."""

    def test_calculate_impact_for_production_service(self, classifier_config):
        """Test impact calculation for production service."""
        classifier = IncidentClassifier(config=classifier_config)

        incident = Incident(
            incident_id=str(uuid4()),
            title="Production outage",
            description="Production service down",
            incident_type=IncidentType.INFRASTRUCTURE,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P1,
            alerts=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=["production", "customer-facing"],
            metadata={"environment": "production"},
        )

        impact = classifier.calculate_business_impact(incident)

        assert impact is not None
        assert "score" in impact
        assert impact["score"] > 0

    def test_calculate_impact_considers_affected_services(self, classifier_config):
        """Test impact calculation considers affected services."""
        classifier = IncidentClassifier(config=classifier_config)

        incident_few_services = Incident(
            incident_id=str(uuid4()),
            title="Limited outage",
            description="Single service affected",
            incident_type=IncidentType.INFRASTRUCTURE,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P2,
            alerts=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=[],
            metadata={"affected_services": ["service-1"]},
        )

        incident_many_services = Incident(
            incident_id=str(uuid4()),
            title="Wide outage",
            description="Multiple services affected",
            incident_type=IncidentType.INFRASTRUCTURE,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P2,
            alerts=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=[],
            metadata={"affected_services": ["service-1", "service-2", "service-3", "service-4"]},
        )

        impact_few = classifier.calculate_business_impact(incident_few_services)
        impact_many = classifier.calculate_business_impact(incident_many_services)

        assert impact_many["score"] > impact_few["score"]

    def test_calculate_impact_considers_duration(self, classifier_config):
        """Test impact calculation considers incident duration."""
        classifier = IncidentClassifier(config=classifier_config)

        incident_short = Incident(
            incident_id=str(uuid4()),
            title="Short incident",
            description="Brief outage",
            incident_type=IncidentType.INFRASTRUCTURE,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P2,
            alerts=[],
            created_at=datetime.utcnow() - timedelta(minutes=5),
            updated_at=datetime.utcnow(),
            tags=[],
            metadata={},
        )

        incident_long = Incident(
            incident_id=str(uuid4()),
            title="Long incident",
            description="Extended outage",
            incident_type=IncidentType.INFRASTRUCTURE,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P2,
            alerts=[],
            created_at=datetime.utcnow() - timedelta(hours=2),
            updated_at=datetime.utcnow(),
            tags=[],
            metadata={},
        )

        impact_short = classifier.calculate_business_impact(incident_short)
        impact_long = classifier.calculate_business_impact(incident_long)

        assert impact_long["score"] >= impact_short["score"]

    def test_calculate_impact_returns_breakdown(self, classifier_config, sample_incident):
        """Test impact calculation returns detailed breakdown."""
        classifier = IncidentClassifier(config=classifier_config)

        impact = classifier.calculate_business_impact(sample_incident)

        assert "score" in impact
        assert "factors" in impact
        assert isinstance(impact["factors"], dict)


class TestSeverityScoring:
    """Test severity score calculation."""

    @pytest.mark.parametrize("severity,expected_weight", [
        ("critical", 1.0),
        ("high", 0.75),
        ("warning", 0.5),
        ("info", 0.25),
    ])
    def test_severity_weights(self, classifier_config, severity, expected_weight):
        """Test severity weights are applied correctly."""
        classifier = IncidentClassifier(config=classifier_config)

        weight = classifier._get_severity_weight(severity)

        assert weight == expected_weight

    def test_unknown_severity_gets_default_weight(self, classifier_config):
        """Test unknown severity gets default weight."""
        classifier = IncidentClassifier(config=classifier_config)

        weight = classifier._get_severity_weight("unknown")

        assert weight >= 0

    def test_aggregate_severity_score(self, classifier_config, multiple_alerts):
        """Test aggregating severity scores from multiple alerts."""
        classifier = IncidentClassifier(config=classifier_config)

        incident = Incident(
            incident_id=str(uuid4()),
            title="Multi-alert incident",
            description="Multiple alerts",
            incident_type=IncidentType.INFRASTRUCTURE,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P3,
            alerts=multiple_alerts,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=[],
            metadata={},
        )

        score = classifier._calculate_aggregate_severity(incident)

        assert 0 <= score <= 1


class TestSeverityLevelDataclass:
    """Test SeverityLevel dataclass."""

    def test_severity_level_creation(self):
        """Test creating SeverityLevel."""
        level = SeverityLevel(
            name="critical",
            weight=1.0,
            description="Critical severity",
            response_time_minutes=15,
            notification_channels=["pagerduty", "slack"],
        )

        assert level.name == "critical"
        assert level.weight == 1.0
        assert level.response_time_minutes == 15

    def test_severity_level_comparison(self):
        """Test SeverityLevel comparison."""
        critical = SeverityLevel(
            name="critical",
            weight=1.0,
            description="Critical",
            response_time_minutes=15,
            notification_channels=[],
        )

        high = SeverityLevel(
            name="high",
            weight=0.75,
            description="High",
            response_time_minutes=30,
            notification_channels=[],
        )

        assert critical.weight > high.weight


class TestTypeMapping:
    """Test incident type mapping from alert sources."""

    @pytest.mark.parametrize("source,expected_type", [
        (AlertSource.GUARDDUTY, IncidentType.SECURITY),
        (AlertSource.WAF, IncidentType.SECURITY),
        (AlertSource.PROMETHEUS, IncidentType.INFRASTRUCTURE),
        (AlertSource.LOKI, IncidentType.APPLICATION),
    ])
    def test_source_to_type_mapping(self, classifier_config, source, expected_type):
        """Test alert source maps to correct incident type."""
        classifier = IncidentClassifier(config=classifier_config)

        alert = Alert(
            alert_id=str(uuid4()),
            title="Test alert",
            description="Test",
            severity="high",
            source=source,
            timestamp=datetime.utcnow(),
            labels={},
            annotations={},
            raw_data={},
        )

        incident = Incident(
            incident_id=str(uuid4()),
            title="Test incident",
            description="Test",
            incident_type=IncidentType.UNKNOWN,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P3,
            alerts=[alert],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=[],
            metadata={},
        )

        classified = classifier.classify(incident)

        assert classified.incident_type == expected_type

    def test_cloudtrail_maps_to_security(self, classifier_config):
        """Test CloudTrail alerts map to security incidents."""
        classifier = IncidentClassifier(config=classifier_config)

        alert = Alert(
            alert_id=str(uuid4()),
            title="CloudTrail alert",
            description="Suspicious API call",
            severity="high",
            source=AlertSource.CLOUDTRAIL,
            timestamp=datetime.utcnow(),
            labels={},
            annotations={},
            raw_data={},
        )

        incident = Incident(
            incident_id=str(uuid4()),
            title="CloudTrail incident",
            description="Test",
            incident_type=IncidentType.UNKNOWN,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P3,
            alerts=[alert],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=[],
            metadata={},
        )

        classified = classifier.classify(incident)

        assert classified.incident_type == IncidentType.SECURITY


class TestClassificationMetadata:
    """Test classification metadata updates."""

    def test_classify_adds_metadata(self, classifier_config, sample_incident):
        """Test classification adds metadata to incident."""
        classifier = IncidentClassifier(config=classifier_config)

        original_metadata = dict(sample_incident.metadata)
        classified = classifier.classify(sample_incident)

        # Should have additional classification metadata
        assert len(classified.metadata) >= len(original_metadata)

    def test_classify_adds_classification_timestamp(self, classifier_config, sample_incident):
        """Test classification adds timestamp to metadata."""
        classifier = IncidentClassifier(config=classifier_config)

        classified = classifier.classify(sample_incident)

        assert "classified_at" in classified.metadata

    def test_classify_preserves_existing_metadata(self, classifier_config):
        """Test classification preserves existing metadata."""
        classifier = IncidentClassifier(config=classifier_config)

        incident = Incident(
            incident_id=str(uuid4()),
            title="Test incident",
            description="Test",
            incident_type=IncidentType.UNKNOWN,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P3,
            alerts=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=[],
            metadata={"custom_field": "custom_value"},
        )

        classified = classifier.classify(incident)

        assert classified.metadata["custom_field"] == "custom_value"
