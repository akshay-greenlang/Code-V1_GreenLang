# OBS-004: Unified Alerting & Notification Platform - Task List

**PRD**: `GreenLang Development/05-Documentation/PRD-OBS-004-Alerting-Notification-Platform.md`
**Status**: IN PROGRESS
**Created**: 2026-02-07

---

## Phase 1: Core SDK + PagerDuty + Opsgenie

### 1.1 Core Module Structure
- [x] Create `greenlang/infrastructure/alerting_service/__init__.py` — public API exports (~80 lines)
- [x] Create `greenlang/infrastructure/alerting_service/config.py` — AlertingConfig dataclass with env defaults (~200 lines)
- [x] Create `greenlang/infrastructure/alerting_service/models.py` — Alert, Notification, EscalationPolicy, OnCallSchedule, enums (~350 lines)
- [x] Create `greenlang/infrastructure/alerting_service/lifecycle.py` — alert state machine (FIRING->ACK->INVESTIGATING->RESOLVED) (~250 lines)
- [x] Create `greenlang/infrastructure/alerting_service/deduplication.py` — fingerprinting, dedup window, correlation (~200 lines)
- [x] Create `greenlang/infrastructure/alerting_service/metrics.py` — 10 Prometheus metrics (~120 lines)

### 1.2 Notification Channels
- [x] Create `greenlang/infrastructure/alerting_service/channels/__init__.py` — channel registry + factory (~50 lines)
- [x] Create `greenlang/infrastructure/alerting_service/channels/base.py` — BaseNotificationChannel ABC (~120 lines)
- [x] Create `greenlang/infrastructure/alerting_service/channels/pagerduty.py` — PagerDuty Events API v2 (trigger/ack/resolve + on-call) (~250 lines)
- [x] Create `greenlang/infrastructure/alerting_service/channels/opsgenie.py` — Opsgenie Alert API v2 (create/ack/close + on-call) (~280 lines)
- [x] Create `greenlang/infrastructure/alerting_service/channels/slack.py` — Slack Block Kit webhooks, severity routing (~250 lines)
- [x] Create `greenlang/infrastructure/alerting_service/channels/email.py` — AWS SES / SMTP email (~200 lines)
- [x] Create `greenlang/infrastructure/alerting_service/channels/teams.py` — Microsoft Teams Adaptive Cards (~220 lines)
- [x] Create `greenlang/infrastructure/alerting_service/channels/webhook.py` — generic HTTP webhook with HMAC-SHA256 (~150 lines)

### 1.3 Routing, Escalation & On-Call
- [x] Create `greenlang/infrastructure/alerting_service/router.py` — AlertRouter (severity/team/service/time routing) (~300 lines)
- [x] Create `greenlang/infrastructure/alerting_service/escalation.py` — EscalationEngine (time-based auto-escalation) (~250 lines)
- [x] Create `greenlang/infrastructure/alerting_service/oncall.py` — OnCallManager (PD + OG schedule lookup, caching) (~300 lines)

### 1.4 Templates
- [x] Create `greenlang/infrastructure/alerting_service/templates/__init__.py` — template module init (~10 lines)
- [x] Create `greenlang/infrastructure/alerting_service/templates/engine.py` — Jinja2 template renderer (~200 lines)
- [x] Create `greenlang/infrastructure/alerting_service/templates/formatters.py` — channel-specific formatters (Block Kit, Adaptive Cards) (~300 lines)

### 1.5 Analytics & Webhook
- [x] Create `greenlang/infrastructure/alerting_service/analytics.py` — MTTA/MTTR, fatigue scoring, reports (~300 lines)
- [x] Create `greenlang/infrastructure/alerting_service/webhook_receiver.py` — Alertmanager webhook endpoint (~200 lines)

### 1.6 API & Setup
- [x] Create `greenlang/infrastructure/alerting_service/api/__init__.py` — API module init (~5 lines)
- [x] Create `greenlang/infrastructure/alerting_service/api/router.py` — REST API 17 endpoints (~400 lines)
- [x] Create `greenlang/infrastructure/alerting_service/setup.py` — configure_alerting(app) (~120 lines)

---

## Phase 2: Deployment Infrastructure

### 2.1 Database Migration
- [x] Create `deployment/database/migrations/sql/V019__alerting_service.sql` — 4 tables, 1 hypertable, 1 continuous aggregate, indexes, permissions

### 2.2 Kubernetes Manifests
- [x] Create `deployment/kubernetes/alerting-service/namespace.yaml` — shared monitoring namespace
- [x] Create `deployment/kubernetes/alerting-service/deployment.yaml` — 2-replica deployment
- [x] Create `deployment/kubernetes/alerting-service/service.yaml` — ClusterIP on port 8080
- [x] Create `deployment/kubernetes/alerting-service/configmap.yaml` — routing rules, escalation policies
- [x] Create `deployment/kubernetes/alerting-service/hpa.yaml` — HPA min 2, max 6, CPU 60%
- [x] Create `deployment/kubernetes/alerting-service/networkpolicy.yaml` — ingress/egress rules
- [x] Create `deployment/kubernetes/alerting-service/servicemonitor.yaml` — Prometheus scrape
- [x] Create `deployment/kubernetes/alerting-service/kustomization.yaml` — Kustomize base

### 2.3 Terraform Module
- [x] Create `deployment/terraform/modules/alerting-integrations/main.tf` — module orchestration
- [x] Create `deployment/terraform/modules/alerting-integrations/variables.tf` — input variables
- [x] Create `deployment/terraform/modules/alerting-integrations/outputs.tf` — outputs
- [x] Create `deployment/terraform/modules/alerting-integrations/pagerduty.tf` — PD provider resources
- [x] Create `deployment/terraform/modules/alerting-integrations/opsgenie.tf` — OG provider resources
- [x] Create `deployment/terraform/modules/alerting-integrations/ssm.tf` — SSM Parameter Store
- [x] Create `deployment/terraform/environments/dev/alerting.tf` — dev environment config
- [x] Create `deployment/terraform/environments/staging/alerting.tf` — staging environment config
- [x] Create `deployment/terraform/environments/prod/alerting.tf` — prod environment config

### 2.4 Monitoring
- [x] Create `deployment/monitoring/dashboards/alerting-service.json` — 20-panel Grafana dashboard
- [x] Create `deployment/monitoring/alerts/alerting-service-alerts.yaml` — 12 PrometheusRule alert rules

### 2.5 Runbooks
- [x] Create `docs/runbooks/alerting-service-down.md` — alerting service failure diagnosis
- [x] Create `docs/runbooks/notification-delivery-failing.md` — channel delivery troubleshooting
- [x] Create `docs/runbooks/pagerduty-integration-down.md` — PD API issues, key rotation

### 2.6 CI/CD
- [x] Create `.github/workflows/alerting-ci.yml` — 6 jobs (lint, unit test, integration, helm, terraform, schema)

---

## Phase 3: Tests

### 3.1 Unit Tests
- [x] Create `tests/unit/alerting_service/__init__.py`
- [x] Create `tests/unit/alerting_service/conftest.py` — shared fixtures
- [x] Create `tests/unit/alerting_service/test_config.py` — 15 tests
- [x] Create `tests/unit/alerting_service/test_models.py` — 20 tests
- [x] Create `tests/unit/alerting_service/test_router.py` — 25 tests
- [x] Create `tests/unit/alerting_service/test_lifecycle.py` — 20 tests
- [x] Create `tests/unit/alerting_service/test_deduplication.py` — 15 tests
- [x] Create `tests/unit/alerting_service/test_escalation.py` — 18 tests
- [x] Create `tests/unit/alerting_service/test_oncall.py` — 12 tests
- [x] Create `tests/unit/alerting_service/test_channels_pagerduty.py` — 15 tests
- [x] Create `tests/unit/alerting_service/test_channels_opsgenie.py` — 15 tests
- [x] Create `tests/unit/alerting_service/test_channels_slack.py` — 12 tests
- [x] Create `tests/unit/alerting_service/test_channels_email.py` — 10 tests
- [x] Create `tests/unit/alerting_service/test_templates.py` — 12 tests
- [x] Create `tests/unit/alerting_service/test_analytics.py` — 15 tests
- [x] Create `tests/unit/alerting_service/test_metrics.py` — 8 tests
- [x] Create `tests/unit/alerting_service/test_webhook_receiver.py` — 12 tests

### 3.2 Integration Tests
- [x] Create `tests/integration/alerting_service/__init__.py`
- [x] Create `tests/integration/alerting_service/conftest.py` — integration fixtures
- [x] Create `tests/integration/alerting_service/test_end_to_end.py` — 12 tests
- [x] Create `tests/integration/alerting_service/test_alertmanager_webhook.py` — 8 tests
- [x] Create `tests/integration/alerting_service/test_channel_delivery.py` — 10 tests

### 3.3 Load Tests
- [x] Create `tests/load/alerting_service/__init__.py`
- [x] Create `tests/load/alerting_service/test_throughput.py` — 6 tests
- [x] Create `tests/load/alerting_service/test_burst.py` — 4 tests

---

## Summary

| Category | Files | Est. Lines |
|----------|-------|-----------|
| Python SDK (23 files) | 23 | ~4,890 |
| K8s Manifests | 8 | ~600 |
| Terraform Module | 9 | ~800 |
| DB Migration | 1 | ~100 |
| Dashboard + Alerts | 2 | ~1,200 |
| Runbooks | 3 | ~900 |
| CI/CD | 1 | ~200 |
| Unit Tests (17 files) | 17 | ~3,500 |
| Integration Tests (5 files) | 5 | ~800 |
| Load Tests (3 files) | 3 | ~400 |
| **TOTAL** | **72** | **~13,390** |
