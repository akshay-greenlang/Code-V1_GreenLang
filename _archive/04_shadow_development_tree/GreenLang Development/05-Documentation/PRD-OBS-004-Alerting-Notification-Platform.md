# PRD-OBS-004: Unified Alerting & Notification Platform

**Component**: OBS-004 - Alerting & Notification Platform (PagerDuty/Opsgenie)
**Priority**: P1 - High
**Status**: Approved
**Version**: 1.0
**Date**: 2026-02-07
**Author**: GreenLang Platform Team
**Depends On**: OBS-001 (Prometheus), OBS-002 (Grafana), OBS-003 (Tracing), SEC-005 (Audit Logging), SEC-010 (Incident Response)

---

## 1. Executive Summary

Build a unified Alerting & Notification Platform that consolidates GreenLang's four fragmented notification implementations into a single, production-grade SDK. The platform adds **Opsgenie** support (currently missing), implements on-call schedule integration, intelligent escalation policies, alert lifecycle management (fire → acknowledge → resolve), MTTA/MTTR analytics, and centralised Jinja2-based notification templates. It wraps the existing Alertmanager HA cluster (OBS-001) with a higher-level application layer that provides programmatic alert management, cross-channel deduplication, and compliance-grade audit trails for all notification activity.

## 2. Current State Assessment

### 2.1 Existing Infrastructure (Do NOT Duplicate)

| Component | Location | Lines | Status |
|-----------|----------|-------|--------|
| Alertmanager HA Cluster | `deployment/helm/prometheus-stack/` | Config | Fully deployed, 2 replicas |
| 32 Alert Rule Files | `deployment/monitoring/alerts/` | ~8,000 | 200+ rules across all services |
| Alertmanager Health Alerts | `deployment/monitoring/alerts/alertmanager-health-alerts.yaml` | 445 | 16 alert rules |
| Alertmanager Dashboard | `deployment/monitoring/dashboards/alertmanager-health.json` | ~500 | 16 panels |
| Active Alerts Dashboard | `deployment/monitoring/dashboards/active-alerts.json` | ~400 | Real-time alert view |
| Incident Response Notifier | `greenlang/infrastructure/incident_response/notifier.py` | 919 | PagerDuty, Slack, Email, SMS |
| Security Scanning Notifier | `greenlang/infrastructure/security_scanning/notifications.py` | 1,034 | Slack, Email, PagerDuty, Teams |
| Legacy Alert Rules Module | `greenlang/monitoring/alerts/alert_rules.py` | 558 | 23 rules, Prometheus/Grafana export |
| Grafana Alert Manager | `greenlang/monitoring/grafana/alert_manager.py` | 150+ | Unified alerting provisioning |
| Grafana Notifiers ConfigMap | `deployment/helm/grafana/templates/configmap-notifiers.yaml` | ~200 | PagerDuty, Slack, Email contacts |

### 2.2 Key Gaps

1. **No Opsgenie Integration**: Zero Opsgenie support across the entire codebase
2. **Fragmented Notification Code**: 4 separate implementations with incompatible APIs
3. **No Unified SDK**: Each service builds its own notification logic
4. **No On-Call Management**: No integration with PagerDuty/Opsgenie schedules
5. **No Escalation Engine**: Manual-only escalation; no automated time-based escalation
6. **No Alert Lifecycle Tracking**: No centralised fire→ack→resolve state machine
7. **No MTTA/MTTR Analytics**: No metrics on alert response times or fatigue
8. **No Deduplication Across Sources**: Alertmanager deduplicates within Prometheus only
9. **No Notification Templates Engine**: Templates are hardcoded strings in each notifier
10. **No Alert Analytics Dashboard**: No visibility into notification delivery or response patterns

## 3. Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Alert Sources                                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────┐ │
│  │ Alertmanager │ │ Application  │ │ Security     │ │ Custom     │ │
│  │ (Prometheus) │ │ Alerts       │ │ Scanners     │ │ Sources    │ │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └─────┬──────┘ │
└─────────┼────────────────┼────────────────┼───────────────┼─────────┘
          │                │                │               │
          ▼                ▼                ▼               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Unified Alerting Service (OBS-004)                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                   Alert Intake & Deduplication                  │ │
│  │  Webhook receiver │ Alertmanager webhook │ API endpoint         │ │
│  └────────────────────────────┬───────────────────────────────────┘ │
│                               │                                     │
│  ┌────────────────────────────▼───────────────────────────────────┐ │
│  │                   Alert Lifecycle Engine                        │ │
│  │  FIRING → ACKNOWLEDGED → INVESTIGATING → RESOLVED              │ │
│  │  Deduplication │ Correlation │ Grouping │ Suppression           │ │
│  └────────────────────────────┬───────────────────────────────────┘ │
│                               │                                     │
│  ┌────────────────────────────▼───────────────────────────────────┐ │
│  │                   Routing & Escalation Engine                   │ │
│  │  Severity routing │ Team routing │ Time-based │ On-call lookup  │ │
│  │  Escalation policies │ Auto-escalate on SLA breach             │ │
│  └────────────────────────────┬───────────────────────────────────┘ │
│                               │                                     │
│  ┌────────────────────────────▼───────────────────────────────────┐ │
│  │                   Notification Channels                         │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │ │
│  │  │PagerDuty │ │ Opsgenie │ │  Slack   │ │  Email   │         │ │
│  │  │Events v2 │ │ Alert v2 │ │ Block Kit│ │ SES/SMTP │         │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘         │ │
│  │  ┌──────────┐ ┌──────────┐                                    │ │
│  │  │  Teams   │ │ Webhook  │                                    │ │
│  │  │ Adaptive │ │ Generic  │                                    │ │
│  │  └──────────┘ └──────────┘                                    │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌──────────────────┐ ┌──────────────────┐ ┌────────────────────┐ │
│  │ Template Engine   │ │ Analytics Engine  │ │ On-Call Manager    │ │
│  │ Jinja2 + Channel │ │ MTTA/MTTR/Fatigue│ │ PD + OG Schedules │ │
│  │ formatters        │ │ TimescaleDB      │ │ Override support   │ │
│  └──────────────────┘ └──────────────────┘ └────────────────────┘ │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Prometheus Metrics + Audit Log                   │  │
│  │  Notification counts, latency, delivery success, MTTA/MTTR   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Selection

| Component | Technology | Version | Justification |
|-----------|-----------|---------|---------------|
| PagerDuty | Events API v2 | Latest | Industry standard; already partially integrated |
| Opsgenie | Alert API v2 | Latest | Atlassian ecosystem; requested by ops team |
| Slack | Block Kit API | Latest | Rich formatting; already configured |
| Email | AWS SES | Latest | Already in use; cost-effective |
| Teams | Adaptive Cards | 1.5 | Already in SEC-007; enterprise requirement |
| Templates | Jinja2 | 3.x | Standard Python templating; powerful |
| Analytics | TimescaleDB | Existing | Already deployed (INFRA-002); time-series native |

### 3.3 Integration Strategy

- **Wraps, doesn't replace** Alertmanager — OBS-004 receives Alertmanager webhooks
- **Wraps, doesn't replace** existing notifiers — provides migration path
- **Backward-compatible** — existing alert rules continue to work unchanged
- **Additive** — adds Opsgenie, on-call, escalation, analytics on top

## 4. Detailed Component Specifications

### 4.1 Unified Python Alerting SDK

**Location**: `greenlang/infrastructure/alerting_service/`

**Files**:

| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `__init__.py` | Public API exports | ~80 |
| `config.py` | AlertingConfig with channel configs, env defaults | ~200 |
| `models.py` | Alert, Notification, EscalationPolicy, OnCallSchedule models | ~350 |
| `router.py` | AlertRouter — severity/team/service/time routing engine | ~300 |
| `lifecycle.py` | Alert lifecycle state machine (fire→ack→resolve) | ~250 |
| `deduplication.py` | Cross-source alert dedup, fingerprinting, correlation | ~200 |
| `escalation.py` | Escalation policy engine, time-based auto-escalation | ~250 |
| `oncall.py` | On-call schedule integration (PagerDuty + Opsgenie APIs) | ~300 |
| `templates/engine.py` | Jinja2 template rendering with channel-specific formatters | ~200 |
| `templates/formatters.py` | Channel-specific formatters (Slack Block Kit, Teams Cards) | ~300 |
| `channels/__init__.py` | Channel registry and factory | ~50 |
| `channels/base.py` | BaseNotificationChannel abstract class | ~120 |
| `channels/pagerduty.py` | PagerDuty Events API v2 integration | ~250 |
| `channels/opsgenie.py` | Opsgenie Alert API v2 integration | ~280 |
| `channels/slack.py` | Slack Incoming Webhooks with Block Kit | ~250 |
| `channels/email.py` | AWS SES / SMTP email sender | ~200 |
| `channels/teams.py` | Microsoft Teams Adaptive Cards | ~220 |
| `channels/webhook.py` | Generic HTTP webhook channel | ~150 |
| `analytics.py` | MTTA/MTTR tracking, alert fatigue metrics, reports | ~300 |
| `metrics.py` | Prometheus metrics for alerting service | ~120 |
| `webhook_receiver.py` | FastAPI webhook endpoint for Alertmanager webhooks | ~200 |
| `api/router.py` | REST API for alert management (15+ endpoints) | ~400 |
| `setup.py` | `configure_alerting(app)` one-liner setup | ~120 |

**Total**: ~23 files, ~4,890 lines estimated

### 4.2 Configuration

```python
@dataclass
class AlertingConfig:
    # Service identification
    service_name: str = "greenlang"
    environment: str = "dev"  # dev, staging, prod
    enabled: bool = True

    # PagerDuty
    pagerduty_enabled: bool = True
    pagerduty_routing_key: str = ""  # Events API v2 routing key
    pagerduty_api_key: str = ""       # REST API key (for on-call lookup)
    pagerduty_service_id: str = ""    # Default service ID

    # Opsgenie
    opsgenie_enabled: bool = True
    opsgenie_api_key: str = ""        # Alert API key
    opsgenie_api_url: str = "https://api.opsgenie.com"
    opsgenie_team: str = ""           # Default team

    # Slack
    slack_enabled: bool = True
    slack_webhook_critical: str = ""   # #platform-alerts-critical
    slack_webhook_warning: str = ""    # #platform-alerts
    slack_webhook_info: str = ""       # #platform-notifications

    # Email
    email_enabled: bool = True
    email_from: str = "alerts@greenlang.io"
    email_smtp_host: str = ""
    email_smtp_port: int = 587
    email_use_ses: bool = True
    email_ses_region: str = "eu-west-1"

    # Teams
    teams_enabled: bool = False
    teams_webhook_url: str = ""

    # Webhook
    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_secret: str = ""

    # Routing
    default_severity_routing: Dict[str, List[str]] = field(default_factory=lambda: {
        "critical": ["pagerduty", "opsgenie", "slack"],
        "warning": ["slack", "email"],
        "info": ["email"],
    })

    # Escalation
    escalation_enabled: bool = True
    escalation_ack_timeout_minutes: int = 15   # Auto-escalate if unacked
    escalation_resolve_timeout_hours: int = 24  # Alert if unresolved

    # Analytics
    analytics_enabled: bool = True
    analytics_retention_days: int = 365

    # Deduplication
    dedup_window_minutes: int = 60  # Group duplicate alerts within window

    # Rate limiting
    rate_limit_per_minute: int = 120
    rate_limit_per_channel_per_minute: int = 60
```

### 4.3 Alert Model

```python
class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

class AlertStatus(str, Enum):
    FIRING = "firing"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class Alert:
    alert_id: str               # UUID
    fingerprint: str            # Dedup key (hash of source + labels)
    source: str                 # "alertmanager", "application", "security"
    name: str                   # Alert name
    severity: AlertSeverity
    status: AlertStatus
    title: str
    description: str
    labels: Dict[str, str]      # Prometheus-style labels
    annotations: Dict[str, str] # Runbook URLs, dashboard links
    tenant_id: Optional[str]
    team: str                   # Owning team
    service: str                # Affected service
    environment: str
    fired_at: datetime
    acknowledged_at: Optional[datetime]
    acknowledged_by: Optional[str]
    resolved_at: Optional[datetime]
    resolved_by: Optional[str]
    escalation_level: int       # 0=initial, 1=escalated, 2=management
    notification_count: int     # How many notifications sent
    runbook_url: Optional[str]
    dashboard_url: Optional[str]
    related_trace_id: Optional[str]  # Trace correlation
```

### 4.4 Notification Channels

#### 4.4.1 PagerDuty (Events API v2)

```python
class PagerDutyChannel(BaseNotificationChannel):
    """PagerDuty Events API v2 integration.

    Supports: trigger, acknowledge, resolve events.
    Maps GreenLang severity → PagerDuty severity (critical/error/warning/info).
    Includes custom details with runbook URL, dashboard URL, trace ID.
    """
    async def send(self, alert: Alert, template: str) -> NotificationResult
    async def acknowledge(self, dedup_key: str) -> NotificationResult
    async def resolve(self, dedup_key: str) -> NotificationResult
    async def get_oncall(self, schedule_id: str) -> List[OnCallUser]
```

#### 4.4.2 Opsgenie (Alert API v2)

```python
class OpsgenieChannel(BaseNotificationChannel):
    """Opsgenie Alert API v2 integration.

    Supports: create, acknowledge, close, add note.
    Maps GreenLang severity → Opsgenie priority (P1-P5).
    Supports team routing, tags, responder assignment.
    """
    async def send(self, alert: Alert, template: str) -> NotificationResult
    async def acknowledge(self, alert_id: str, user: str) -> NotificationResult
    async def close(self, alert_id: str, user: str) -> NotificationResult
    async def get_oncall(self, schedule_id: str) -> List[OnCallUser]
```

#### 4.4.3 Slack (Block Kit)

```python
class SlackChannel(BaseNotificationChannel):
    """Slack Incoming Webhooks with Block Kit formatting.

    Severity-based channel routing:
      critical → #platform-alerts-critical
      warning  → #platform-alerts
      info     → #platform-notifications
    Rich formatting with: header, context, actions, dividers.
    """
    async def send(self, alert: Alert, template: str) -> NotificationResult
```

#### 4.4.4 Email (SES/SMTP)

```python
class EmailChannel(BaseNotificationChannel):
    """AWS SES or SMTP email notification channel.

    HTML email with responsive template.
    Team-based recipient routing.
    """
    async def send(self, alert: Alert, template: str) -> NotificationResult
```

#### 4.4.5 Microsoft Teams (Adaptive Cards)

```python
class TeamsChannel(BaseNotificationChannel):
    """Microsoft Teams Incoming Webhook with Adaptive Cards.

    Rich card formatting with severity colours, action buttons.
    """
    async def send(self, alert: Alert, template: str) -> NotificationResult
```

#### 4.4.6 Generic Webhook

```python
class WebhookChannel(BaseNotificationChannel):
    """Generic HTTP webhook for custom integrations.

    HMAC-SHA256 signature on payload. Configurable headers.
    """
    async def send(self, alert: Alert, template: str) -> NotificationResult
```

### 4.5 Routing & Escalation Engine

**Routing Rules** (evaluated in priority order):

1. **Explicit override**: Alert labels contain `routing.channel=pagerduty`
2. **Team routing**: Map team → channel config (e.g., `security-team` → PagerDuty)
3. **Service routing**: Map service → channel config
4. **Severity routing**: Default severity → channels mapping
5. **Time-based routing**: Business hours vs off-hours routing

**Escalation Policies**:

```python
@dataclass
class EscalationPolicy:
    name: str
    steps: List[EscalationStep]

@dataclass
class EscalationStep:
    delay_minutes: int          # Wait before escalating
    channels: List[str]         # Channels to notify at this level
    oncall_schedule_id: str     # PagerDuty/Opsgenie schedule to look up
    notify_users: List[str]     # Explicit user list
    repeat: int = 1             # Repeat this step N times
```

Default escalation for critical alerts:
- **Step 1** (0 min): Notify on-call via PagerDuty + Slack
- **Step 2** (15 min): Re-notify on-call + notify team lead
- **Step 3** (30 min): Escalate to engineering manager + Opsgenie
- **Step 4** (60 min): Escalate to VP Engineering + Email

### 4.6 On-Call Management

```python
class OnCallManager:
    """Fetch on-call schedules from PagerDuty and Opsgenie."""

    async def get_current_oncall(self, schedule_id: str, provider: str) -> OnCallUser
    async def get_oncall_schedule(self, schedule_id: str, provider: str) -> Schedule
    async def list_schedules(self, provider: str) -> List[Schedule]
    async def override_oncall(self, schedule_id: str, user_id: str, ...) -> Override
```

### 4.7 Alert Analytics

**Metrics tracked in TimescaleDB + Prometheus**:

| Metric | Type | Description |
|--------|------|-------------|
| `gl_alert_notifications_total` | Counter | Total notifications sent by channel, severity, status |
| `gl_alert_notification_duration_seconds` | Histogram | Notification delivery latency |
| `gl_alert_notification_failures_total` | Counter | Failed notification attempts |
| `gl_alert_mtta_seconds` | Histogram | Mean Time To Acknowledge |
| `gl_alert_mttr_seconds` | Histogram | Mean Time To Resolve |
| `gl_alert_active_total` | Gauge | Currently active alerts by severity |
| `gl_alert_escalations_total` | Counter | Escalation events by level |
| `gl_alert_dedup_total` | Counter | Deduplicated (suppressed) alerts |
| `gl_alert_fatigue_score` | Gauge | Alert fatigue index (alerts/hour/team) |
| `gl_alert_oncall_lookups_total` | Counter | On-call schedule lookups |

**Analytics Reports**:
- Weekly MTTA/MTTR by team
- Alert fatigue index (alerts per on-call rotation)
- Top 10 noisiest alerts
- Notification delivery success rate by channel
- Escalation frequency by policy

### 4.8 API Endpoints

**Location**: `greenlang/infrastructure/alerting_service/api/router.py`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/alerts` | Create a new alert |
| GET | `/api/v1/alerts` | List alerts (filter by status, severity, team) |
| GET | `/api/v1/alerts/{id}` | Get alert details |
| PATCH | `/api/v1/alerts/{id}/acknowledge` | Acknowledge an alert |
| PATCH | `/api/v1/alerts/{id}/resolve` | Resolve an alert |
| PATCH | `/api/v1/alerts/{id}/escalate` | Manually escalate |
| PATCH | `/api/v1/alerts/{id}/suppress` | Suppress/snooze |
| POST | `/api/v1/alerts/{id}/note` | Add a note to alert |
| POST | `/api/v1/alerts/webhook/alertmanager` | Alertmanager webhook receiver |
| GET | `/api/v1/alerts/analytics/mtta` | MTTA report by team |
| GET | `/api/v1/alerts/analytics/mttr` | MTTR report by team |
| GET | `/api/v1/alerts/analytics/fatigue` | Alert fatigue report |
| GET | `/api/v1/alerts/analytics/top-noisy` | Top noisiest alerts |
| GET | `/api/v1/alerts/oncall` | Current on-call for all schedules |
| GET | `/api/v1/alerts/oncall/{schedule_id}` | On-call for specific schedule |
| GET | `/api/v1/alerts/channels/health` | Channel health status |
| POST | `/api/v1/alerts/test` | Send a test notification |

### 4.9 Alertmanager Webhook Integration

OBS-004 receives Alertmanager webhook notifications and processes them through the unified pipeline:

```yaml
# Alertmanager config update (add to existing routing)
receivers:
  - name: greenlang-alerting-service
    webhook_configs:
      - url: http://alerting-service:8080/api/v1/alerts/webhook/alertmanager
        send_resolved: true
        max_alerts: 0  # Send all
```

### 4.10 Kubernetes Manifests

**Location**: `deployment/kubernetes/alerting-service/`

| File | Description |
|------|-------------|
| `namespace.yaml` | Namespace `monitoring` (shared) |
| `deployment.yaml` | 2-replica deployment with health checks |
| `service.yaml` | ClusterIP service on port 8080 |
| `configmap.yaml` | Routing rules, escalation policies, templates |
| `hpa.yaml` | HPA (min 2, max 6, CPU 60%) |
| `networkpolicy.yaml` | Ingress from Alertmanager, Grafana; egress to PD/OG/Slack |
| `servicemonitor.yaml` | Prometheus scrape config |
| `kustomization.yaml` | Kustomize base |

### 4.11 Terraform Module

**Location**: `deployment/terraform/modules/alerting-integrations/`

| File | Description |
|------|-------------|
| `main.tf` | PagerDuty service + escalation policy + Opsgenie team + API keys in SSM |
| `variables.tf` | Input variables (PD key, OG key, Slack webhooks) |
| `outputs.tf` | Integration IDs, webhook URLs |
| `pagerduty.tf` | PagerDuty provider: service, integration, escalation policy |
| `opsgenie.tf` | Opsgenie provider: team, integration, escalation, schedule |
| `ssm.tf` | AWS SSM Parameter Store for secrets |

**Environment configs**: `deployment/terraform/environments/{dev,staging,prod}/alerting.tf`

### 4.12 Database Migration

**Location**: `deployment/database/migrations/sql/V019__alerting_service.sql`

```sql
-- Alert tracking table (TimescaleDB hypertable)
CREATE TABLE alerting.alerts (
    alert_id UUID PRIMARY KEY,
    fingerprint VARCHAR(64) NOT NULL,
    source VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'firing',
    title TEXT NOT NULL,
    description TEXT,
    labels JSONB NOT NULL DEFAULT '{}',
    annotations JSONB NOT NULL DEFAULT '{}',
    tenant_id VARCHAR(50),
    team VARCHAR(100),
    service VARCHAR(100),
    environment VARCHAR(20),
    fired_at TIMESTAMPTZ NOT NULL,
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(100),
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(100),
    escalation_level INTEGER DEFAULT 0,
    notification_count INTEGER DEFAULT 0,
    runbook_url TEXT,
    dashboard_url TEXT,
    related_trace_id VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Notification log table (hypertable for time-series analytics)
CREATE TABLE alerting.notification_log (
    id BIGSERIAL,
    alert_id UUID REFERENCES alerting.alerts(alert_id),
    channel VARCHAR(30) NOT NULL,
    status VARCHAR(20) NOT NULL,  -- sent, failed, rate_limited
    recipient VARCHAR(255),
    duration_ms INTEGER,
    response_code INTEGER,
    error_message TEXT,
    sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
SELECT create_hypertable('alerting.notification_log', 'sent_at');

-- Escalation history
CREATE TABLE alerting.escalation_log (
    id BIGSERIAL,
    alert_id UUID REFERENCES alerting.alerts(alert_id),
    from_level INTEGER NOT NULL,
    to_level INTEGER NOT NULL,
    reason VARCHAR(100),
    escalated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- On-call cache
CREATE TABLE alerting.oncall_cache (
    schedule_id VARCHAR(100) PRIMARY KEY,
    provider VARCHAR(20) NOT NULL,
    oncall_user_id VARCHAR(100),
    oncall_user_name VARCHAR(255),
    oncall_user_email VARCHAR(255),
    valid_until TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- MTTA/MTTR continuous aggregates
CREATE MATERIALIZED VIEW alerting.alert_response_metrics_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', fired_at) AS bucket,
    team,
    severity,
    COUNT(*) AS alert_count,
    AVG(EXTRACT(EPOCH FROM (acknowledged_at - fired_at))) AS avg_mtta_seconds,
    AVG(EXTRACT(EPOCH FROM (resolved_at - fired_at))) AS avg_mttr_seconds,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (acknowledged_at - fired_at))) AS p95_mtta,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (resolved_at - fired_at))) AS p95_mttr
FROM alerting.alerts
WHERE fired_at IS NOT NULL
GROUP BY bucket, team, severity
WITH NO DATA;

-- Indexes
CREATE INDEX idx_alerts_status ON alerting.alerts(status);
CREATE INDEX idx_alerts_severity ON alerting.alerts(severity);
CREATE INDEX idx_alerts_team ON alerting.alerts(team);
CREATE INDEX idx_alerts_fingerprint ON alerting.alerts(fingerprint);
CREATE INDEX idx_alerts_fired_at ON alerting.alerts(fired_at DESC);
CREATE INDEX idx_notification_log_alert ON alerting.notification_log(alert_id);
CREATE INDEX idx_notification_log_channel ON alerting.notification_log(channel);

-- Permissions
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA alerting TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA alerting TO greenlang_app;
```

### 4.13 Monitoring & Alerting

#### Grafana Dashboard

**Location**: `deployment/monitoring/dashboards/alerting-service.json`

**Panels** (20):
1. Active alerts by severity (stat)
2. Notifications sent/min by channel (time series)
3. Notification success rate by channel (gauge)
4. Notification latency P50/P95/P99 (time series)
5. MTTA by team (bar chart)
6. MTTR by team (bar chart)
7. Alert fatigue score by team (gauge)
8. Escalation events/hour (time series)
9. Deduplication rate (stat)
10. Top 10 noisiest alerts (table)
11. Alert lifecycle flow (state timeline)
12. On-call lookup latency (time series)
13. Channel health status (status map)
14. Alerts by source (pie chart)
15. Alerts by service (bar chart)
16. Notification delivery failures (time series)
17. Rate limiting events (time series)
18. Alert volume trend (time series)
19. Resolution time distribution (histogram)
20. Weekly MTTA/MTTR trend (time series)

#### Alert Rules

**Location**: `deployment/monitoring/alerts/alerting-service-alerts.yaml`

| Alert | Severity | Condition |
|-------|----------|-----------|
| `AlertingServiceDown` | critical | No healthy instances for 5m |
| `NotificationDeliveryFailing` | critical | Failure rate > 10% for 5m |
| `PagerDutyIntegrationDown` | critical | PD delivery failures > 0 for 10m |
| `OpsgenieIntegrationDown` | critical | OG delivery failures > 0 for 10m |
| `SlackIntegrationDown` | warning | Slack failures > 5% for 10m |
| `HighMTTA` | warning | Team MTTA > 15min for 30m |
| `AlertNotAcknowledged` | warning | Critical alert unacked > 15m |
| `EscalationTriggered` | info | Any escalation event |
| `AlertFatigueHigh` | warning | Fatigue score > 80 for 1h |
| `NotificationRateLimited` | warning | Rate limiting active for 5m |
| `OnCallLookupFailing` | warning | On-call lookups failing > 50% |
| `HighAlertVolume` | info | > 100 alerts/hour for 30m |

#### Runbooks

**Location**: `docs/runbooks/`

| Runbook | Description |
|---------|-------------|
| `alerting-service-down.md` | Diagnose alerting service failures |
| `notification-delivery-failing.md` | Channel delivery troubleshooting |
| `pagerduty-integration-down.md` | PagerDuty API issues, key rotation |

### 4.14 CI/CD Pipeline

**Location**: `.github/workflows/alerting-ci.yml`

**Jobs**:
1. **lint**: Python linting + YAML validation
2. **test-unit**: Run unit tests with coverage
3. **test-integration**: Integration tests with mock channels
4. **helm-lint**: Lint Helm templates
5. **terraform-validate**: Validate Terraform modules
6. **schema-validate**: Dashboard JSON + alert YAML validation

## 5. Testing Requirements

### 5.1 Unit Tests

**Location**: `tests/unit/alerting_service/`

| Test File | Tests (est.) | Coverage |
|-----------|-------------|----------|
| `test_config.py` | 15 | Config defaults, env parsing, validation |
| `test_models.py` | 20 | Alert model, severity mapping, serialization |
| `test_router.py` | 25 | Routing rules, severity routing, team routing |
| `test_lifecycle.py` | 20 | State transitions, timestamps, validation |
| `test_deduplication.py` | 15 | Fingerprinting, dedup window, correlation |
| `test_escalation.py` | 18 | Policy evaluation, timing, auto-escalation |
| `test_oncall.py` | 12 | Schedule lookup, caching, fallback |
| `test_channels_pagerduty.py` | 15 | PD event creation, ack, resolve |
| `test_channels_opsgenie.py` | 15 | OG alert creation, ack, close |
| `test_channels_slack.py` | 12 | Block Kit formatting, webhook delivery |
| `test_channels_email.py` | 10 | SES/SMTP, HTML templating |
| `test_templates.py` | 12 | Jinja2 rendering, channel formatters |
| `test_analytics.py` | 15 | MTTA/MTTR calculation, fatigue scoring |
| `test_metrics.py` | 8 | Prometheus metric recording |
| `test_webhook_receiver.py` | 12 | Alertmanager webhook parsing |

**Total**: ~224 unit tests

### 5.2 Integration Tests

**Location**: `tests/integration/alerting_service/`

| Test File | Tests (est.) | Coverage |
|-----------|-------------|----------|
| `test_end_to_end.py` | 12 | Full alert flow: create→route→notify→ack→resolve |
| `test_alertmanager_webhook.py` | 8 | Parse real Alertmanager webhook payloads |
| `test_channel_delivery.py` | 10 | Mock HTTP delivery for all channels |
| `conftest.py` | N/A | Fixtures, mock servers |

**Total**: ~30 integration tests

### 5.3 Load Tests

**Location**: `tests/load/alerting_service/`

| Test File | Tests (est.) | Coverage |
|-----------|-------------|----------|
| `test_throughput.py` | 6 | 1000 alerts/min sustained |
| `test_burst.py` | 4 | 100 simultaneous alerts |

**Total**: ~10 load tests

## 6. Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Notification delivery latency | < 2s P99 | Prometheus histogram |
| Notification success rate | > 99.5% | Prometheus counter |
| Channel count | 6 (PD, OG, Slack, Email, Teams, Webhook) | Code review |
| MTTA tracking accuracy | ±5 seconds | Integration test |
| Alert deduplication rate | > 30% reduction in noise | Analytics dashboard |
| Escalation accuracy | 100% of SLA breaches escalated | Integration test |
| API endpoint count | 17 | Code review |
| Test count | 260+ (224 unit + 30 integration + 10 load) | pytest count |
| Code coverage | 85%+ | pytest-cov report |

## 7. Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| `httpx` | ~=0.27 | Async HTTP client for channel APIs |
| `jinja2` | ~=3.1 | Template rendering engine |
| `pydantic` | ~=2.6 | Alert model validation |

**No new dependencies required** — httpx, jinja2, and pydantic are already in the project.

## 8. Rollout Plan

### Phase 1 (Week 1): Core SDK + PagerDuty + Opsgenie
- Build unified alerting SDK with config, models, lifecycle
- Implement PagerDuty and Opsgenie channels
- Deploy webhook receiver for Alertmanager
- Unit tests for core components

### Phase 2 (Week 2): Routing + Escalation + Templates
- Build routing engine and escalation policies
- Implement Slack, Email, Teams, Webhook channels
- Build Jinja2 template engine
- Configure Alertmanager webhook forwarding

### Phase 3 (Week 3): Analytics + On-Call + Dashboard
- Implement MTTA/MTTR analytics
- Build on-call schedule integration
- Deploy Grafana dashboard and alert rules
- Integration and load tests

### Phase 4 (Week 4): Production + Migration
- Deploy to production
- Migrate SEC-010 notifier to use unified SDK
- Migrate SEC-007 notifier to use unified SDK
- Complete runbook documentation

---

**Approved By**: Platform Engineering Team
**Review Date**: 2026-02-07
