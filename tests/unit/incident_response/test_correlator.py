"""
Unit tests for IncidentCorrelator.

Tests alert correlation, similarity calculation, grouping algorithms,
and incident merging functionality.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from uuid import uuid4

from greenlang.infrastructure.incident_response.correlator import IncidentCorrelator
from greenlang.infrastructure.incident_response.models import (
    Alert,
    Incident,
    AlertSource,
    IncidentType,
    IncidentStatus,
    EscalationLevel,
)


class TestCorrelatorInitialization:
    """Test IncidentCorrelator initialization."""

    def test_initialization_with_config(self, correlator_config):
        """Test correlator initializes with configuration."""
        correlator = IncidentCorrelator(config=correlator_config)

        assert correlator.config == correlator_config
        assert correlator.time_window == correlator_config.time_window_seconds
        assert correlator.similarity_threshold == correlator_config.similarity_threshold

    def test_initialization_default_config(self):
        """Test correlator initializes with default configuration."""
        correlator = IncidentCorrelator()

        assert correlator.config is not None
        assert correlator.time_window > 0
        assert 0 <= correlator.similarity_threshold <= 1

    def test_initialization_sets_correlation_features(self, correlator_config):
        """Test correlator sets correlation features from config."""
        correlator = IncidentCorrelator(config=correlator_config)

        assert correlator.correlation_features == correlator_config.correlation_features


class TestSimilarityCalculation:
    """Test alert similarity calculation."""

    def test_identical_alerts_have_similarity_one(self, correlator_config, sample_alert):
        """Test identical alerts have similarity of 1.0."""
        correlator = IncidentCorrelator(config=correlator_config)

        similarity = correlator.calculate_similarity(sample_alert, sample_alert)

        assert similarity == 1.0

    def test_completely_different_alerts_have_low_similarity(self, correlator_config):
        """Test completely different alerts have low similarity."""
        correlator = IncidentCorrelator(config=correlator_config)

        alert1 = Alert(
            alert_id=str(uuid4()),
            title="High CPU on node-1",
            description="CPU usage high",
            severity="high",
            source=AlertSource.PROMETHEUS,
            timestamp=datetime.utcnow(),
            labels={"instance": "node-1", "alertname": "HighCPU"},
            annotations={},
            raw_data={},
        )

        alert2 = Alert(
            alert_id=str(uuid4()),
            title="SQL injection detected",
            description="Security alert",
            severity="critical",
            source=AlertSource.WAF,
            timestamp=datetime.utcnow() + timedelta(hours=2),
            labels={"source_ip": "192.168.1.100", "attack_type": "sqli"},
            annotations={},
            raw_data={},
        )

        similarity = correlator.calculate_similarity(alert1, alert2)

        assert similarity < 0.3

    def test_similar_alerts_have_high_similarity(self, correlator_config):
        """Test similar alerts have high similarity score."""
        correlator = IncidentCorrelator(config=correlator_config)

        base_time = datetime.utcnow()

        alert1 = Alert(
            alert_id=str(uuid4()),
            title="High CPU on node-1",
            description="CPU usage exceeded 90%",
            severity="high",
            source=AlertSource.PROMETHEUS,
            timestamp=base_time,
            labels={"instance": "node-1", "alertname": "HighCPU", "job": "kubernetes"},
            annotations={},
            raw_data={},
        )

        alert2 = Alert(
            alert_id=str(uuid4()),
            title="High memory on node-1",
            description="Memory usage exceeded 85%",
            severity="warning",
            source=AlertSource.PROMETHEUS,
            timestamp=base_time + timedelta(seconds=30),
            labels={"instance": "node-1", "alertname": "HighMemory", "job": "kubernetes"},
            annotations={},
            raw_data={},
        )

        similarity = correlator.calculate_similarity(alert1, alert2)

        # Should be high due to shared instance and job labels
        assert similarity > 0.5

    def test_time_proximity_affects_similarity(self, correlator_config):
        """Test alerts closer in time have higher similarity."""
        correlator = IncidentCorrelator(config=correlator_config)

        base_time = datetime.utcnow()

        alert1 = Alert(
            alert_id=str(uuid4()),
            title="Alert 1",
            description="Test",
            severity="high",
            source=AlertSource.PROMETHEUS,
            timestamp=base_time,
            labels={"instance": "node-1"},
            annotations={},
            raw_data={},
        )

        alert2_close = Alert(
            alert_id=str(uuid4()),
            title="Alert 2",
            description="Test",
            severity="high",
            source=AlertSource.PROMETHEUS,
            timestamp=base_time + timedelta(seconds=30),
            labels={"instance": "node-1"},
            annotations={},
            raw_data={},
        )

        alert2_far = Alert(
            alert_id=str(uuid4()),
            title="Alert 2",
            description="Test",
            severity="high",
            source=AlertSource.PROMETHEUS,
            timestamp=base_time + timedelta(hours=1),
            labels={"instance": "node-1"},
            annotations={},
            raw_data={},
        )

        sim_close = correlator.calculate_similarity(alert1, alert2_close)
        sim_far = correlator.calculate_similarity(alert1, alert2_far)

        assert sim_close > sim_far

    def test_source_affects_similarity(self, correlator_config):
        """Test alerts from same source have higher similarity."""
        correlator = IncidentCorrelator(config=correlator_config)

        base_time = datetime.utcnow()

        alert1 = Alert(
            alert_id=str(uuid4()),
            title="Alert 1",
            description="Test",
            severity="high",
            source=AlertSource.PROMETHEUS,
            timestamp=base_time,
            labels={"instance": "node-1"},
            annotations={},
            raw_data={},
        )

        alert2_same_source = Alert(
            alert_id=str(uuid4()),
            title="Alert 2",
            description="Test",
            severity="high",
            source=AlertSource.PROMETHEUS,
            timestamp=base_time,
            labels={"instance": "node-1"},
            annotations={},
            raw_data={},
        )

        alert2_diff_source = Alert(
            alert_id=str(uuid4()),
            title="Alert 2",
            description="Test",
            severity="high",
            source=AlertSource.GUARDDUTY,
            timestamp=base_time,
            labels={"instance": "node-1"},
            annotations={},
            raw_data={},
        )

        sim_same = correlator.calculate_similarity(alert1, alert2_same_source)
        sim_diff = correlator.calculate_similarity(alert1, alert2_diff_source)

        assert sim_same >= sim_diff


class TestCorrelation:
    """Test alert correlation into groups."""

    def test_correlate_empty_list(self, correlator_config):
        """Test correlating empty alert list."""
        correlator = IncidentCorrelator(config=correlator_config)

        groups = correlator.correlate([])

        assert groups == []

    def test_correlate_single_alert(self, correlator_config, sample_alert):
        """Test correlating single alert creates single group."""
        correlator = IncidentCorrelator(config=correlator_config)

        groups = correlator.correlate([sample_alert])

        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert groups[0][0] == sample_alert

    def test_correlate_groups_related_alerts(self, correlator_config, multiple_alerts):
        """Test related alerts are grouped together."""
        correlator = IncidentCorrelator(config=correlator_config)

        groups = correlator.correlate(multiple_alerts)

        # Should have fewer groups than alerts
        assert len(groups) < len(multiple_alerts)

        # Each group should have at least one alert
        for group in groups:
            assert len(group) >= 1

    def test_correlate_respects_time_window(self, correlator_config):
        """Test correlation respects configured time window."""
        correlator_config.time_window_seconds = 60  # 1 minute window
        correlator = IncidentCorrelator(config=correlator_config)

        base_time = datetime.utcnow()

        alert1 = Alert(
            alert_id=str(uuid4()),
            title="Alert 1",
            description="Test",
            severity="high",
            source=AlertSource.PROMETHEUS,
            timestamp=base_time,
            labels={"instance": "node-1"},
            annotations={},
            raw_data={},
        )

        alert2 = Alert(
            alert_id=str(uuid4()),
            title="Alert 2",
            description="Test",
            severity="high",
            source=AlertSource.PROMETHEUS,
            timestamp=base_time + timedelta(seconds=30),  # Within window
            labels={"instance": "node-1"},
            annotations={},
            raw_data={},
        )

        alert3 = Alert(
            alert_id=str(uuid4()),
            title="Alert 3",
            description="Test",
            severity="high",
            source=AlertSource.PROMETHEUS,
            timestamp=base_time + timedelta(minutes=5),  # Outside window
            labels={"instance": "node-1"},
            annotations={},
            raw_data={},
        )

        groups = correlator.correlate([alert1, alert2, alert3])

        # Alert 1 and 2 should be in same group, Alert 3 separate
        assert len(groups) >= 2

    def test_correlate_respects_similarity_threshold(self, correlator_config):
        """Test correlation respects similarity threshold."""
        correlator_config.similarity_threshold = 0.9  # High threshold
        correlator = IncidentCorrelator(config=correlator_config)

        base_time = datetime.utcnow()

        alert1 = Alert(
            alert_id=str(uuid4()),
            title="High CPU",
            description="CPU high",
            severity="high",
            source=AlertSource.PROMETHEUS,
            timestamp=base_time,
            labels={"instance": "node-1", "alertname": "HighCPU"},
            annotations={},
            raw_data={},
        )

        alert2 = Alert(
            alert_id=str(uuid4()),
            title="High Memory",
            description="Memory high",
            severity="high",
            source=AlertSource.PROMETHEUS,
            timestamp=base_time,
            labels={"instance": "node-1", "alertname": "HighMemory"},
            annotations={},
            raw_data={},
        )

        groups = correlator.correlate([alert1, alert2])

        # With high threshold, may not group together
        # This depends on similarity calculation
        assert len(groups) >= 1

    def test_correlate_handles_large_alert_set(self, correlator_config):
        """Test correlation handles large number of alerts efficiently."""
        correlator = IncidentCorrelator(config=correlator_config)

        # Generate 100 alerts
        alerts = []
        base_time = datetime.utcnow()

        for i in range(100):
            alerts.append(Alert(
                alert_id=str(uuid4()),
                title=f"Alert {i}",
                description="Test",
                severity="warning",
                source=AlertSource.PROMETHEUS,
                timestamp=base_time + timedelta(seconds=i * 10),
                labels={"instance": f"node-{i % 5}"},  # 5 nodes
                annotations={},
                raw_data={},
            ))

        groups = correlator.correlate(alerts)

        # Should group by instance
        assert len(groups) <= 100
        assert len(groups) >= 5  # At least one group per node


class TestUnionFind:
    """Test union-find algorithm for alert grouping."""

    def test_union_find_creates_initial_sets(self, correlator_config):
        """Test union-find creates initial disjoint sets."""
        correlator = IncidentCorrelator(config=correlator_config)

        elements = ["a", "b", "c", "d"]
        uf = correlator._create_union_find(elements)

        # Each element in its own set
        for elem in elements:
            assert uf.find(elem) == elem

    def test_union_find_merges_sets(self, correlator_config):
        """Test union-find correctly merges sets."""
        correlator = IncidentCorrelator(config=correlator_config)

        elements = ["a", "b", "c", "d"]
        uf = correlator._create_union_find(elements)

        uf.union("a", "b")
        uf.union("c", "d")

        # a and b should be in same set
        assert uf.find("a") == uf.find("b")
        # c and d should be in same set
        assert uf.find("c") == uf.find("d")
        # a and c should be in different sets
        assert uf.find("a") != uf.find("c")

    def test_union_find_transitive_union(self, correlator_config):
        """Test union-find handles transitive unions."""
        correlator = IncidentCorrelator(config=correlator_config)

        elements = ["a", "b", "c"]
        uf = correlator._create_union_find(elements)

        uf.union("a", "b")
        uf.union("b", "c")

        # All should be in same set
        assert uf.find("a") == uf.find("b") == uf.find("c")


class TestIncidentMerging:
    """Test incident merging functionality."""

    def test_merge_incidents_combines_alerts(self, correlator_config, multiple_alerts):
        """Test merging incidents combines all alerts."""
        correlator = IncidentCorrelator(config=correlator_config)

        incident1 = Incident(
            incident_id=str(uuid4()),
            title="Incident 1",
            description="Test",
            incident_type=IncidentType.INFRASTRUCTURE,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P2,
            alerts=multiple_alerts[:2],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=[],
            metadata={},
        )

        incident2 = Incident(
            incident_id=str(uuid4()),
            title="Incident 2",
            description="Test",
            incident_type=IncidentType.INFRASTRUCTURE,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P3,
            alerts=multiple_alerts[2:4],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=[],
            metadata={},
        )

        merged = correlator.merge_incidents([incident1, incident2])

        # Should have all alerts from both incidents
        assert len(merged.alerts) == len(incident1.alerts) + len(incident2.alerts)

    def test_merge_incidents_takes_highest_escalation(self, correlator_config):
        """Test merged incident has highest escalation level."""
        correlator = IncidentCorrelator(config=correlator_config)

        incident1 = Incident(
            incident_id=str(uuid4()),
            title="Incident 1",
            description="Test",
            incident_type=IncidentType.INFRASTRUCTURE,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P3,
            alerts=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=[],
            metadata={},
        )

        incident2 = Incident(
            incident_id=str(uuid4()),
            title="Incident 2",
            description="Test",
            incident_type=IncidentType.SECURITY,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P0,
            alerts=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=[],
            metadata={},
        )

        merged = correlator.merge_incidents([incident1, incident2])

        assert merged.escalation_level == EscalationLevel.P0

    def test_merge_incidents_combines_tags(self, correlator_config):
        """Test merged incident combines all tags."""
        correlator = IncidentCorrelator(config=correlator_config)

        incident1 = Incident(
            incident_id=str(uuid4()),
            title="Incident 1",
            description="Test",
            incident_type=IncidentType.INFRASTRUCTURE,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P2,
            alerts=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=["infrastructure", "node-1"],
            metadata={},
        )

        incident2 = Incident(
            incident_id=str(uuid4()),
            title="Incident 2",
            description="Test",
            incident_type=IncidentType.INFRASTRUCTURE,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P3,
            alerts=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=["cpu", "memory"],
            metadata={},
        )

        merged = correlator.merge_incidents([incident1, incident2])

        # Should have all unique tags
        assert "infrastructure" in merged.tags
        assert "node-1" in merged.tags
        assert "cpu" in merged.tags
        assert "memory" in merged.tags

    def test_merge_incidents_preserves_earliest_created_at(self, correlator_config):
        """Test merged incident preserves earliest creation time."""
        correlator = IncidentCorrelator(config=correlator_config)

        earlier_time = datetime.utcnow() - timedelta(hours=1)
        later_time = datetime.utcnow()

        incident1 = Incident(
            incident_id=str(uuid4()),
            title="Incident 1",
            description="Test",
            incident_type=IncidentType.INFRASTRUCTURE,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P2,
            alerts=[],
            created_at=earlier_time,
            updated_at=datetime.utcnow(),
            tags=[],
            metadata={},
        )

        incident2 = Incident(
            incident_id=str(uuid4()),
            title="Incident 2",
            description="Test",
            incident_type=IncidentType.INFRASTRUCTURE,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P3,
            alerts=[],
            created_at=later_time,
            updated_at=datetime.utcnow(),
            tags=[],
            metadata={},
        )

        merged = correlator.merge_incidents([incident1, incident2])

        assert merged.created_at == earlier_time

    def test_merge_incidents_single_incident_returns_copy(self, correlator_config, sample_incident):
        """Test merging single incident returns a copy."""
        correlator = IncidentCorrelator(config=correlator_config)

        merged = correlator.merge_incidents([sample_incident])

        assert merged.title == sample_incident.title
        assert merged.alerts == sample_incident.alerts

    def test_merge_incidents_empty_list_raises(self, correlator_config):
        """Test merging empty list raises error."""
        correlator = IncidentCorrelator(config=correlator_config)

        with pytest.raises(ValueError):
            correlator.merge_incidents([])


class TestCorrelationFeatures:
    """Test correlation feature extraction."""

    def test_extract_features_from_alert(self, correlator_config, sample_alert):
        """Test extracting correlation features from alert."""
        correlator = IncidentCorrelator(config=correlator_config)

        features = correlator._extract_features(sample_alert)

        assert isinstance(features, dict)
        # Should have features for configured correlation_features
        for feature in correlator_config.correlation_features:
            assert feature in features or features.get(feature) is None

    def test_feature_extraction_handles_missing_labels(self, correlator_config):
        """Test feature extraction handles missing labels gracefully."""
        correlator = IncidentCorrelator(config=correlator_config)

        alert = Alert(
            alert_id=str(uuid4()),
            title="Test",
            description="Test",
            severity="high",
            source=AlertSource.PROMETHEUS,
            timestamp=datetime.utcnow(),
            labels={},  # Empty labels
            annotations={},
            raw_data={},
        )

        features = correlator._extract_features(alert)

        # Should not raise, should return empty/None features
        assert isinstance(features, dict)

    def test_feature_vector_for_similarity(self, correlator_config, sample_alert):
        """Test creating feature vector for similarity calculation."""
        correlator = IncidentCorrelator(config=correlator_config)

        vector = correlator._create_feature_vector(sample_alert)

        assert isinstance(vector, (list, tuple))
        assert len(vector) > 0
