# OBS-002: Configure Grafana Dashboards - Task List

**PRD:** `GreenLang Development/05-Documentation/PRD-OBS-002-Grafana-Dashboards.md`
**Status:** IN PROGRESS
**Created:** 2026-02-07
**Priority:** P0 - Critical

---

## Phase 1: Helm Chart & Deployment Infrastructure

### 1.1 Helm Chart
- [x] Create `deployment/helm/grafana/Chart.yaml`
- [x] Create `deployment/helm/grafana/values.yaml` (base values)
- [x] Create `deployment/helm/grafana/values-dev.yaml`
- [x] Create `deployment/helm/grafana/values-staging.yaml`
- [x] Create `deployment/helm/grafana/values-prod.yaml`
- [x] Create `deployment/helm/grafana/templates/_helpers.tpl`
- [x] Create `deployment/helm/grafana/templates/configmap-grafana-ini.yaml`
- [x] Create `deployment/helm/grafana/templates/configmap-datasources.yaml`
- [x] Create `deployment/helm/grafana/templates/configmap-dashboard-providers.yaml`
- [x] Create `deployment/helm/grafana/templates/configmap-notifiers.yaml`
- [x] Create `deployment/helm/grafana/templates/deployment.yaml`
- [x] Create `deployment/helm/grafana/templates/headless-service.yaml`
- [x] Create `deployment/helm/grafana/templates/service.yaml`
- [x] Create `deployment/helm/grafana/templates/ingress.yaml`
- [x] Create `deployment/helm/grafana/templates/hpa.yaml`
- [x] Create `deployment/helm/grafana/templates/pdb.yaml`
- [x] Create `deployment/helm/grafana/templates/networkpolicy.yaml`
- [x] Create `deployment/helm/grafana/templates/serviceaccount.yaml`
- [x] Create `deployment/helm/grafana/templates/servicemonitor.yaml`
- [x] Create `deployment/helm/grafana/templates/secret.yaml`
- [x] Create `deployment/helm/grafana/templates/pvc.yaml`
- [x] Create `deployment/helm/grafana/templates/deployment-renderer.yaml`
- [x] Create `deployment/helm/grafana/templates/service-renderer.yaml`
- [x] Create `deployment/helm/grafana/templates/NOTES.txt`

### 1.2 Terraform Module
- [x] Create `deployment/terraform/modules/grafana/main.tf`
- [x] Create `deployment/terraform/modules/grafana/variables.tf`
- [x] Create `deployment/terraform/modules/grafana/outputs.tf`
- [x] Create `deployment/terraform/modules/grafana/iam.tf`
- [x] Create `deployment/terraform/modules/grafana/s3.tf`
- [x] Create `deployment/terraform/modules/grafana/security-groups.tf`

### 1.3 Environment Terraform Configs
- [x] Create `deployment/terraform/environments/dev/grafana.tf`
- [x] Create `deployment/terraform/environments/staging/grafana.tf`
- [x] Create `deployment/terraform/environments/prod/grafana.tf`

## Phase 2: Dashboard Provisioning & Organization

### 2.1 K8s Dashboard ConfigMaps
- [x] Create `deployment/kubernetes/grafana/namespace.yaml`
- [x] Create `deployment/kubernetes/grafana/configmap-dashboards-executive.yaml`
- [x] Create `deployment/kubernetes/grafana/configmap-dashboards-infrastructure.yaml`
- [x] Create `deployment/kubernetes/grafana/configmap-dashboards-datastores.yaml`
- [x] Create `deployment/kubernetes/grafana/configmap-dashboards-observability.yaml`
- [x] Create `deployment/kubernetes/grafana/configmap-dashboards-security.yaml`
- [x] Create `deployment/kubernetes/grafana/configmap-dashboards-applications.yaml`
- [x] Create `deployment/kubernetes/grafana/configmap-dashboards-alerts.yaml`
- [x] Create `deployment/kubernetes/grafana/networkpolicy.yaml`
- [x] Create `deployment/kubernetes/grafana/servicemonitor.yaml`
- [x] Create `deployment/kubernetes/grafana/poddisruptionbudget.yaml`
- [x] Create `deployment/kubernetes/grafana/kustomization.yaml`

### 2.2 New Dashboard JSONs
- [x] Create `deployment/monitoring/dashboards/platform-overview.json`
- [x] Create `deployment/monitoring/dashboards/grafana-health.json`
- [x] Create `deployment/monitoring/dashboards/grafana-usage.json`
- [x] Create `deployment/monitoring/dashboards/security-posture.json`
- [x] Create `deployment/monitoring/dashboards/active-alerts.json`
- [x] Create `deployment/monitoring/dashboards/application-health.json`

## Phase 3: Python SDK

### 3.1 SDK Core
- [x] Create `greenlang/monitoring/grafana/__init__.py`
- [x] Create `greenlang/monitoring/grafana/models.py`
- [x] Create `greenlang/monitoring/grafana/client.py`
- [x] Create `greenlang/monitoring/grafana/dashboard_builder.py`
- [x] Create `greenlang/monitoring/grafana/panel_builder.py`
- [x] Create `greenlang/monitoring/grafana/folder_manager.py`
- [x] Create `greenlang/monitoring/grafana/datasource_manager.py`
- [x] Create `greenlang/monitoring/grafana/alert_manager.py`
- [x] Create `greenlang/monitoring/grafana/provisioning.py`

## Phase 4: Monitoring & Alerting

### 4.1 Alert Rules
- [x] Create `deployment/monitoring/alerts/grafana-alerts.yaml`

### 4.2 Runbooks
- [x] Create `docs/runbooks/grafana-down.md`
- [x] Create `docs/runbooks/grafana-high-memory.md`
- [x] Create `docs/runbooks/grafana-dashboard-slow.md`

## Phase 5: CI/CD & Testing

### 5.1 CI/CD
- [x] Create `.github/workflows/grafana-dashboard-ci.yml`

### 5.2 Tests
- [x] Create `tests/unit/monitoring/__init__.py`
- [x] Create `tests/unit/monitoring/test_grafana_client.py`
- [x] Create `tests/unit/monitoring/test_dashboard_builder.py`
- [x] Create `tests/unit/monitoring/test_panel_builder.py`
- [x] Create `tests/unit/monitoring/test_folder_manager.py`
- [x] Create `tests/integration/monitoring/test_grafana_integration.py`
- [x] Create `tests/load/monitoring/test_grafana_load.py`

---

## Summary

| Category | Files | Status |
|----------|-------|--------|
| Helm Chart | 24 | COMPLETE |
| Terraform Module | 6 | COMPLETE |
| Environment Configs | 3 | COMPLETE |
| K8s Manifests | 12 | COMPLETE |
| New Dashboards | 6 | COMPLETE |
| Python SDK | 9 | COMPLETE |
| Alert Rules | 1 | COMPLETE |
| Runbooks | 3 | COMPLETE |
| CI/CD Workflow | 1 | COMPLETE |
| Tests | 7 | COMPLETE |
| **TOTAL** | **72** | **COMPLETE** |
