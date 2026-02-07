"""
Test fixtures for incident_response module.

Provides mock incidents, alerts, playbooks, and configuration
for comprehensive unit testing of incident response components.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from greenlang.infrastructure.incident_response.models import (
    Alert,
    Incident,
    IncidentType,
    IncidentStatus,
    EscalationLevel,
    AlertSource,
    PlaybookStep,
    PlaybookExecution,
    PlaybookStatus,
    TimelineEvent,
    PostMortem,
)
from greenlang.infrastructure.incident_response.config import (
    IncidentResponseConfig,
    DetectorConfig,
    CorrelatorConfig,
    ClassifierConfig,
    PlaybookConfig,
)


# -----------------------------------------------------------------------------
# Alert Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_alert_data() -> Dict[str, Any]:
    """Return raw alert data for testing."""
    return {
        "alert_id": str(uuid4()),
        "title": "High CPU utilization detected",
        "description": "CPU usage exceeded 90% for 5 minutes",
        "severity": "high",
        "source": "prometheus",
        "timestamp": datetime.utcnow().isoformat(),
        "labels": {
            "alertname": "HighCPUUsage",
            "instance": "node-1",
            "job": "kubernetes-nodes",
        },
        "annotations": {
            "summary": "CPU usage is above 90%",
            "runbook_url": "https://wiki.example.com/cpu-alert",
        },
    }


@pytest.fixture
def sample_alert(sample_alert_data) -> Alert:
    """Create a sample Alert instance."""
    return Alert(
        alert_id=sample_alert_data["alert_id"],
        title=sample_alert_data["title"],
        description=sample_alert_data["description"],
        severity=sample_alert_data["severity"],
        source=AlertSource.PROMETHEUS,
        timestamp=datetime.utcnow(),
        labels=sample_alert_data["labels"],
        annotations=sample_alert_data["annotations"],
        raw_data=sample_alert_data,
    )


@pytest.fixture
def multiple_alerts() -> List[Alert]:
    """Create multiple alerts for correlation testing."""
    base_time = datetime.utcnow()
    alerts = []

    # Alert cluster 1: Related infrastructure alerts
    alerts.append(Alert(
        alert_id=str(uuid4()),
        title="High memory usage on node-1",
        description="Memory usage exceeded 85%",
        severity="warning",
        source=AlertSource.PROMETHEUS,
        timestamp=base_time,
        labels={"instance": "node-1", "alertname": "HighMemory"},
        annotations={},
        raw_data={},
    ))
    alerts.append(Alert(
        alert_id=str(uuid4()),
        title="High CPU usage on node-1",
        description="CPU usage exceeded 90%",
        severity="high",
        source=AlertSource.PROMETHEUS,
        timestamp=base_time + timedelta(seconds=30),
        labels={"instance": "node-1", "alertname": "HighCPU"},
        annotations={},
        raw_data={},
    ))
    alerts.append(Alert(
        alert_id=str(uuid4()),
        title="Pod restarts on node-1",
        description="Pod greenlang-api restarted 3 times",
        severity="warning",
        source=AlertSource.PROMETHEUS,
        timestamp=base_time + timedelta(seconds=60),
        labels={"instance": "node-1", "pod": "greenlang-api"},
        annotations={},
        raw_data={},
    ))

    # Alert cluster 2: Security alerts
    alerts.append(Alert(
        alert_id=str(uuid4()),
        title="Suspicious API access pattern",
        description="Unusual number of 401 errors from IP 10.0.0.5",
        severity="high",
        source=AlertSource.GUARDDUTY,
        timestamp=base_time + timedelta(minutes=5),
        labels={"source_ip": "10.0.0.5", "type": "unauthorized_access"},
        annotations={},
        raw_data={},
    ))
    alerts.append(Alert(
        alert_id=str(uuid4()),
        title="Brute force attempt detected",
        description="Multiple failed login attempts from IP 10.0.0.5",
        severity="critical",
        source=AlertSource.GUARDDUTY,
        timestamp=base_time + timedelta(minutes=5, seconds=30),
        labels={"source_ip": "10.0.0.5", "type": "brute_force"},
        annotations={},
        raw_data={},
    ))

    # Isolated alert
    alerts.append(Alert(
        alert_id=str(uuid4()),
        title="Disk space low on backup-server",
        description="Disk usage at 95%",
        severity="warning",
        source=AlertSource.PROMETHEUS,
        timestamp=base_time + timedelta(minutes=10),
        labels={"instance": "backup-server", "alertname": "LowDisk"},
        annotations={},
        raw_data={},
    ))

    return alerts


@pytest.fixture
def security_alert() -> Alert:
    """Create a security-focused alert."""
    return Alert(
        alert_id=str(uuid4()),
        title="Potential SQL injection detected",
        description="WAF blocked suspicious request with SQL patterns",
        severity="critical",
        source=AlertSource.WAF,
        timestamp=datetime.utcnow(),
        labels={
            "rule_id": "sql_injection_attempt",
            "source_ip": "192.168.1.100",
            "path": "/api/v1/users",
            "method": "POST",
        },
        annotations={
            "blocked": "true",
            "attack_type": "sql_injection",
        },
        raw_data={
            "request_body": "username=admin'--",
            "user_agent": "curl/7.68.0",
        },
    )


# -----------------------------------------------------------------------------
# Incident Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_incident(sample_alert) -> Incident:
    """Create a sample Incident instance."""
    return Incident(
        incident_id=str(uuid4()),
        title="Service degradation on node-1",
        description="Multiple resource alerts triggered on node-1",
        incident_type=IncidentType.INFRASTRUCTURE,
        status=IncidentStatus.OPEN,
        escalation_level=EscalationLevel.P2,
        alerts=[sample_alert],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        assigned_to=None,
        tags=["infrastructure", "node-1", "resource-exhaustion"],
        metadata={
            "affected_services": ["greenlang-api", "worker-1"],
            "estimated_impact": "partial_degradation",
        },
    )


@pytest.fixture
def critical_incident(security_alert) -> Incident:
    """Create a critical security incident."""
    return Incident(
        incident_id=str(uuid4()),
        title="Active SQL injection attack",
        description="Multiple SQL injection attempts detected from single source",
        incident_type=IncidentType.SECURITY,
        status=IncidentStatus.OPEN,
        escalation_level=EscalationLevel.P0,
        alerts=[security_alert],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        assigned_to="security-team",
        tags=["security", "sql-injection", "attack", "critical"],
        metadata={
            "attack_source": "192.168.1.100",
            "attack_type": "sql_injection",
            "blocked": True,
            "requires_forensics": True,
        },
    )


@pytest.fixture
def incident_with_timeline(sample_incident) -> Incident:
    """Create incident with timeline events."""
    base_time = sample_incident.created_at
    sample_incident.timeline = [
        TimelineEvent(
            event_id=str(uuid4()),
            timestamp=base_time,
            event_type="incident_created",
            description="Incident automatically created from correlated alerts",
            actor="system",
            metadata={},
        ),
        TimelineEvent(
            event_id=str(uuid4()),
            timestamp=base_time + timedelta(minutes=2),
            event_type="escalated",
            description="Escalated from P3 to P2 due to additional alerts",
            actor="system",
            metadata={"previous_level": "P3", "new_level": "P2"},
        ),
        TimelineEvent(
            event_id=str(uuid4()),
            timestamp=base_time + timedelta(minutes=5),
            event_type="assigned",
            description="Assigned to platform-team",
            actor="oncall-bot",
            metadata={"assignee": "platform-team"},
        ),
        TimelineEvent(
            event_id=str(uuid4()),
            timestamp=base_time + timedelta(minutes=10),
            event_type="playbook_started",
            description="Started playbook: node_resource_remediation",
            actor="platform-team",
            metadata={"playbook_id": "node_resource_remediation"},
        ),
    ]
    return sample_incident


# -----------------------------------------------------------------------------
# Playbook Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_playbook_steps() -> List[PlaybookStep]:
    """Create sample playbook steps."""
    return [
        PlaybookStep(
            step_id="step_1",
            name="Collect diagnostic data",
            description="Gather system metrics and logs",
            action_type="collect_diagnostics",
            parameters={"target": "node-1", "duration_minutes": 5},
            timeout_seconds=300,
            retry_count=2,
            order=1,
        ),
        PlaybookStep(
            step_id="step_2",
            name="Identify root cause",
            description="Analyze collected data",
            action_type="analyze_data",
            parameters={"analysis_type": "resource_exhaustion"},
            timeout_seconds=60,
            retry_count=0,
            order=2,
        ),
        PlaybookStep(
            step_id="step_3",
            name="Apply remediation",
            description="Restart affected pods",
            action_type="restart_pods",
            parameters={"selector": "app=greenlang-api", "namespace": "production"},
            timeout_seconds=120,
            retry_count=1,
            order=3,
        ),
        PlaybookStep(
            step_id="step_4",
            name="Verify recovery",
            description="Check service health",
            action_type="health_check",
            parameters={"endpoint": "/health", "expected_status": 200},
            timeout_seconds=60,
            retry_count=3,
            order=4,
        ),
    ]


@pytest.fixture
def sample_playbook_execution(sample_incident, sample_playbook_steps) -> PlaybookExecution:
    """Create a sample playbook execution."""
    return PlaybookExecution(
        execution_id=str(uuid4()),
        playbook_id="node_resource_remediation",
        incident_id=sample_incident.incident_id,
        status=PlaybookStatus.RUNNING,
        steps=sample_playbook_steps,
        current_step=1,
        started_at=datetime.utcnow(),
        completed_at=None,
        executed_by="platform-team",
        results={},
    )


@pytest.fixture
def completed_playbook_execution(sample_playbook_execution) -> PlaybookExecution:
    """Create a completed playbook execution."""
    sample_playbook_execution.status = PlaybookStatus.COMPLETED
    sample_playbook_execution.current_step = 4
    sample_playbook_execution.completed_at = datetime.utcnow()
    sample_playbook_execution.results = {
        "step_1": {"status": "success", "duration_ms": 4500},
        "step_2": {"status": "success", "root_cause": "memory_leak"},
        "step_3": {"status": "success", "pods_restarted": 3},
        "step_4": {"status": "success", "health_check_passed": True},
    }
    return sample_playbook_execution


# -----------------------------------------------------------------------------
# Configuration Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def detector_config() -> DetectorConfig:
    """Create detector configuration."""
    return DetectorConfig(
        prometheus_url="http://prometheus:9090",
        loki_url="http://loki:3100",
        guardduty_enabled=True,
        cloudtrail_enabled=True,
        poll_interval_seconds=30,
        alert_batch_size=100,
    )


@pytest.fixture
def correlator_config() -> CorrelatorConfig:
    """Create correlator configuration."""
    return CorrelatorConfig(
        time_window_seconds=300,
        similarity_threshold=0.7,
        min_alerts_for_incident=2,
        max_alerts_per_incident=50,
        correlation_features=["instance", "source_ip", "alertname", "service"],
    )


@pytest.fixture
def classifier_config() -> ClassifierConfig:
    """Create classifier configuration."""
    return ClassifierConfig(
        severity_weights={
            "critical": 1.0,
            "high": 0.75,
            "warning": 0.5,
            "info": 0.25,
        },
        type_mappings={
            "guardduty": IncidentType.SECURITY,
            "waf": IncidentType.SECURITY,
            "prometheus": IncidentType.INFRASTRUCTURE,
            "loki": IncidentType.APPLICATION,
        },
        escalation_thresholds={
            EscalationLevel.P0: 0.9,
            EscalationLevel.P1: 0.7,
            EscalationLevel.P2: 0.5,
            EscalationLevel.P3: 0.0,
        },
    )


@pytest.fixture
def playbook_config() -> PlaybookConfig:
    """Create playbook configuration."""
    return PlaybookConfig(
        playbook_directory="/etc/greenlang/playbooks",
        max_concurrent_executions=5,
        default_timeout_seconds=3600,
        enable_auto_remediation=True,
        require_approval_for=["production_rollback", "data_deletion"],
    )


@pytest.fixture
def incident_response_config(
    detector_config,
    correlator_config,
    classifier_config,
    playbook_config,
) -> IncidentResponseConfig:
    """Create full incident response configuration."""
    return IncidentResponseConfig(
        detector=detector_config,
        correlator=correlator_config,
        classifier=classifier_config,
        playbook=playbook_config,
        notification_channels=["slack", "pagerduty", "email"],
        retention_days=90,
        enable_metrics=True,
    )


# -----------------------------------------------------------------------------
# Mock Service Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_prometheus_client():
    """Create mock Prometheus client."""
    client = AsyncMock()
    client.query.return_value = {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [
                {
                    "metric": {"__name__": "ALERTS", "alertname": "HighCPU"},
                    "value": [1609459200, "1"],
                }
            ],
        },
    }
    client.query_range.return_value = {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [],
        },
    }
    return client


@pytest.fixture
def mock_loki_client():
    """Create mock Loki client."""
    client = AsyncMock()
    client.query.return_value = {
        "status": "success",
        "data": {
            "resultType": "streams",
            "result": [
                {
                    "stream": {"level": "error", "app": "greenlang-api"},
                    "values": [
                        ["1609459200000000000", "Error processing request"],
                    ],
                }
            ],
        },
    }
    return client


@pytest.fixture
def mock_guardduty_client():
    """Create mock GuardDuty client."""
    client = MagicMock()
    client.list_findings.return_value = {
        "FindingIds": ["finding-1", "finding-2"],
    }
    client.get_findings.return_value = {
        "Findings": [
            {
                "Id": "finding-1",
                "Type": "UnauthorizedAccess:IAMUser/MaliciousIPCaller",
                "Severity": 8.0,
                "Title": "API called from malicious IP",
                "Description": "An API was called from a known malicious IP address",
                "CreatedAt": datetime.utcnow().isoformat(),
            },
        ],
    }
    return client


@pytest.fixture
def mock_slack_client():
    """Create mock Slack client for notifications."""
    client = AsyncMock()
    client.post_message.return_value = {"ok": True, "ts": "1234567890.123456"}
    client.update_message.return_value = {"ok": True}
    return client


@pytest.fixture
def mock_pagerduty_client():
    """Create mock PagerDuty client for escalations."""
    client = AsyncMock()
    client.create_incident.return_value = {
        "incident": {
            "id": "PD-12345",
            "status": "triggered",
            "urgency": "high",
        }
    }
    client.resolve_incident.return_value = {"incident": {"status": "resolved"}}
    return client


@pytest.fixture
def mock_database():
    """Create mock database for incident persistence."""
    db = AsyncMock()
    db.save_incident.return_value = True
    db.get_incident.return_value = None
    db.list_incidents.return_value = []
    db.update_incident.return_value = True
    db.save_playbook_execution.return_value = True
    return db


# -----------------------------------------------------------------------------
# API Test Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_auth_token() -> str:
    """Create mock authentication token."""
    return "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0LXVzZXIiLCJyb2xlcyI6WyJpbmNpZGVudC1yZXNwb25kZXIiXX0.test"


@pytest.fixture
def auth_headers(mock_auth_token) -> Dict[str, str]:
    """Create authentication headers for API tests."""
    return {"Authorization": mock_auth_token}


@pytest.fixture
def incident_responder_headers() -> Dict[str, str]:
    """Create headers for incident responder role."""
    return {
        "Authorization": "Bearer test-incident-responder-token",
        "X-User-Id": "test-user",
        "X-User-Roles": "incident-responder,viewer",
    }


@pytest.fixture
def security_admin_headers() -> Dict[str, str]:
    """Create headers for security admin role."""
    return {
        "Authorization": "Bearer test-security-admin-token",
        "X-User-Id": "security-admin",
        "X-User-Roles": "security-admin,incident-responder,viewer",
    }


# -----------------------------------------------------------------------------
# Post-Mortem Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_post_mortem(sample_incident) -> PostMortem:
    """Create a sample post-mortem document."""
    return PostMortem(
        post_mortem_id=str(uuid4()),
        incident_id=sample_incident.incident_id,
        title=f"Post-Mortem: {sample_incident.title}",
        summary="Service degradation due to memory leak in API service",
        timeline=[
            {
                "time": "14:00 UTC",
                "event": "First alert triggered for high memory usage",
            },
            {
                "time": "14:05 UTC",
                "event": "Pod restarts began",
            },
            {
                "time": "14:10 UTC",
                "event": "Incident created and team paged",
            },
            {
                "time": "14:25 UTC",
                "event": "Root cause identified as memory leak",
            },
            {
                "time": "14:35 UTC",
                "event": "Hotfix deployed",
            },
            {
                "time": "14:45 UTC",
                "event": "Service fully recovered",
            },
        ],
        root_cause="Memory leak in request handler caused by unclosed database connections",
        contributing_factors=[
            "Missing connection pool timeout configuration",
            "No memory limit alerts at 70% threshold",
            "Delayed detection due to noisy monitoring",
        ],
        impact={
            "duration_minutes": 45,
            "affected_users": 150,
            "error_rate_increase": "15%",
            "revenue_impact": None,
        },
        action_items=[
            {
                "action": "Add connection pool timeout",
                "owner": "backend-team",
                "due_date": "2025-02-01",
                "status": "completed",
            },
            {
                "action": "Add 70% memory warning alert",
                "owner": "platform-team",
                "due_date": "2025-02-05",
                "status": "in_progress",
            },
            {
                "action": "Review and reduce alert noise",
                "owner": "sre-team",
                "due_date": "2025-02-10",
                "status": "pending",
            },
        ],
        lessons_learned=[
            "Connection pooling defaults are not suitable for production load",
            "Earlier memory alerts would have reduced detection time",
        ],
        created_at=datetime.utcnow(),
        created_by="platform-team-lead",
        status="published",
    )
