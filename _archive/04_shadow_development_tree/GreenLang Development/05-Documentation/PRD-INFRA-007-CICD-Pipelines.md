# PRD-INFRA-007: CI/CD Pipelines (GitHub Actions)

## GreenLang Climate OS | Infrastructure Component

| Field | Value |
|-------|-------|
| **PRD ID** | INFRA-007 |
| **Component** | CI/CD Pipelines (GitHub Actions) |
| **Priority** | P0 - Critical |
| **Status** | In Development |
| **Author** | GreenLang Infrastructure Team |
| **Created** | 2026-02-04 |
| **Dependencies** | INFRA-001 (K8s), INFRA-002 (PostgreSQL), INFRA-003 (Redis), INFRA-004 (S3), INFRA-006 (Kong) |

---

## 1. Executive Summary

GreenLang Climate OS requires a production-grade CI/CD pipeline infrastructure that automates building, testing, securing, and deploying all platform components. The platform already has **15 active GitHub Actions workflows** covering core CI, security scanning, deployment, and release orchestration. INFRA-007 consolidates this existing infrastructure, fills identified gaps, and establishes a unified pipeline standard with DORA metrics tracking, reusable composite actions, automated environment promotion, and comprehensive observability.

---

## 2. Current State Assessment

### 2.1 Existing Workflows (15 Active)

| # | Workflow | File | Triggers | Purpose |
|---|----------|------|----------|---------|
| 1 | CI Main | `ci-main.yml` | push/PR main,master,develop | Lint, unit tests (3 OS x 3 Python), integration, build, CI gate |
| 2 | PR Validation | `pr-validation.yml` | PR main,master,develop | Agent-specific tests, core library, coverage, determinism |
| 3 | Release Orchestration | `release-orchestration.yml` | tag v*, workflow_dispatch | Version, build, Docker, Sigstore signing, PyPI, GitHub Release |
| 4 | Deploy Environments | `deploy-environments.yml` | push main, workflow_dispatch | K8s deployment to dev/staging/prod with health checks |
| 5 | Unified Security Scan | `unified-security-scan.yml` | PR, daily schedule | Trivy, Snyk, Bandit, Gitleaks, License, SBOM |
| 6 | Pre-commit | `pre-commit.yml` | PR main,master | 30+ hooks: security, quality, IaC, docs |
| 7 | Code Governance | `code-governance.yml` | PR main,master | OPA policy, magic numbers, spec validation, determinism |
| 8 | Acceptance Golden Tests | `acceptance-golden-tests.yml` | PR, weekly | Golden tests, pack validation, performance benchmarks |
| 9 | Agents CI | `agents-ci.yml` | PR, push | Matrix testing for changed agents |
| 10 | Infra Deploy | `infra-deploy.yml` | push main, workflow_dispatch | Terraform plan/apply, K8s deploy, blue-green |
| 11 | Infra Validate | `infra-validate.yml` | PR paths deployment/** | Terraform fmt/lint, tfsec, Checkov, cost estimation |
| 12 | Database Migrations | `database-migrations.yml` | PR paths deployment/database/** | Flyway validation, TimescaleDB testing |
| 13 | Docs Build | `docs-build.yml` | PR paths docs/** | MkDocs build, link validation, GitHub Pages |
| 14 | Scheduled Maintenance | `scheduled-maintenance.yml` | cron daily/weekly | Security scans, dependency updates, benchmarks |
| 15 | Smoke Test | `smoke-test.yml` | workflow_dispatch | Post-release package validation |

### 2.2 Supporting Infrastructure

| Category | Status | Details |
|----------|--------|---------|
| Pre-commit hooks | Mature | 30+ hooks across 9 categories |
| Dockerfiles | Mature | 95+ Dockerfiles across platform |
| Helm charts | Mature | 10+ charts with dev/staging/prod values |
| Terraform modules | Mature | 20+ modules with 3 environments |
| Docker Compose | Mature | 45+ compose files for local dev |
| Makefiles | Mature | 100+ targets across all projects |
| Security scanning | Mature | Trivy, Snyk, Bandit, Gitleaks, tfsec, Checkov |
| Release signing | Mature | Sigstore + Cosign |
| SBOM generation | Mature | CycloneDX + SPDX formats |
| Dependabot | Configured | pip, docker, github-actions, npm |

### 2.3 Identified Gaps

| # | Gap | Impact | Priority |
|---|-----|--------|----------|
| G-001 | No CODEOWNERS file | No automated code review routing | P0 |
| G-002 | No reusable composite actions | Workflow duplication, maintenance burden | P0 |
| G-003 | No automated environment promotion | Manual promotion dev -> staging -> prod | P0 |
| G-004 | No rollback workflow | Manual rollback via kubectl | P1 |
| G-005 | No Helm chart CI | Charts not linted/tested in CI | P1 |
| G-006 | No DORA metrics tracking | No deployment frequency/lead time/MTTR visibility | P1 |
| G-007 | No pipeline observability dashboard | No visibility into CI/CD health | P1 |
| G-008 | No dependency-review-action | PR dependency changes not reviewed | P2 |
| G-009 | No Terraform plan PR comments | TF changes not visible in PR review | P2 |
| G-010 | No ArgoCD sync workflow | GitOps sync not triggered from CI | P2 |
| G-011 | No unified notification strategy | Scattered Slack webhook usage | P2 |
| G-012 | No branch protection automation | Manual branch rule configuration | P2 |
| G-013 | No performance regression detection | Benchmarks exist but no automated gates | P3 |
| G-014 | No CI status badges | README lacks pipeline status | P3 |

---

## 3. Architecture

### 3.1 Pipeline Architecture

```
                         GreenLang CI/CD Pipeline Architecture
  ============================================================================

  Developer Workflow:
  +-----------+    +-------------+    +------------+    +------------------+
  | Pre-commit| -> | PR Created  | -> | CI Gate    | -> | Review Required  |
  | 30+ hooks |    | Auto-checks |    | All green  |    | CODEOWNERS route |
  +-----------+    +-------------+    +------------+    +------------------+
                                                                 |
  ============================================================================
  Merge to main:
       |
       v
  +----------+    +----------+    +-----------+    +----------+    +--------+
  | Build &  | -> | Security | -> | Deploy    | -> | Smoke    | -> | DORA   |
  | Test     |    | Gate     |    | to Dev    |    | Tests    |    | Record |
  +----------+    +----------+    +-----------+    +----------+    +--------+
                                                                       |
  ============================================================================
  Promotion Pipeline:
       |
       v
  +----------+    +----------+    +-----------+    +----------+    +--------+
  | Promote  | -> | Deploy   | -> | Integration| -> | Approve  | -> | Deploy |
  | Staging  |    | Staging  |    | Tests     |    | Prod     |    | Prod   |
  +----------+    +----------+    +-----------+    +----------+    +--------+
       |                                                                |
       v                                                                v
  +----------+                                                   +----------+
  | ArgoCD   |                                                   | Rollback |
  | Sync     |                                                   | Ready    |
  +----------+                                                   +----------+

  ============================================================================
  Release Pipeline:
  +--------+    +--------+    +--------+    +--------+    +--------+
  | Tag    | -> | Build  | -> | Sign   | -> | Publish| -> | Verify |
  | v*     |    | Multi  |    | SBOM   |    | PyPI + |    | Smoke  |
  |        |    | OS/Py  |    | Attest |    | Docker |    | Tests  |
  +--------+    +--------+    +--------+    +--------+    +--------+
```

### 3.2 Reusable Composite Actions

```
.github/actions/
  setup-greenlang/          # Python + GreenLang environment setup
    action.yml
  setup-k8s/                # kubectl + Helm + kubeconfig
    action.yml
  security-gate/            # Unified security scanning gate
    action.yml
  docker-build-push/        # Multi-arch Docker build + push + sign
    action.yml
  notify/                   # Unified Slack/Teams notification
    action.yml
  dora-record/              # DORA metrics recording
    action.yml
```

### 3.3 File Structure

```
.github/
  CODEOWNERS                          # Code ownership mapping
  actions/
    setup-greenlang/action.yml        # Composite: Python + GreenLang setup
    setup-k8s/action.yml              # Composite: K8s tooling setup
    security-gate/action.yml          # Composite: Security scan gate
    docker-build-push/action.yml      # Composite: Docker build/push/sign
    notify/action.yml                 # Composite: Unified notifications
    dora-record/action.yml            # Composite: DORA metrics recording
  workflows/
    promote-environment.yml           # NEW: Automated env promotion
    rollback.yml                      # NEW: Automated rollback
    helm-ci.yml                       # NEW: Helm chart linting/testing
    dependency-review.yml             # NEW: PR dependency review
    dora-metrics.yml                  # NEW: DORA metrics aggregation
    terraform-plan-comment.yml        # NEW: TF plan as PR comment

deployment/
  monitoring/
    dashboards/
      cicd-pipeline.json              # NEW: Grafana CI/CD dashboard
    alerts/
      cicd-alerts.yaml                # NEW: Pipeline health alerts
```

---

## 4. Technical Requirements

### TR-001: CODEOWNERS File

| Field | Value |
|-------|-------|
| **Priority** | P0 |
| **Gap** | G-001 |

**Specification:**
- Map all top-level directories to responsible teams
- Map critical paths (security, deployment, database) to specialized reviewers
- Require minimum 1 approval from code owners for all PRs
- Cover: `greenlang/`, `deployment/`, `.github/`, `tests/`, `applications/`, `scripts/`

### TR-002: Reusable Composite Actions

| Field | Value |
|-------|-------|
| **Priority** | P0 |
| **Gap** | G-002 |

**Specification:**

**setup-greenlang/action.yml:**
- Input: python-version (default 3.11), install-extras (default "test,dev")
- Steps: checkout, setup-python with cache, pip install, version display
- Output: python-path, greenlang-version

**setup-k8s/action.yml:**
- Input: cluster-env (dev/staging/prod), install-helm (default true)
- Steps: setup-kubectl, setup-helm, configure kubeconfig from secrets
- Output: kubeconfig-path, cluster-name

**security-gate/action.yml:**
- Input: scan-type (full/quick), fail-on (critical/high/medium)
- Steps: Trivy filesystem scan, pip-audit, Bandit SAST, Gitleaks
- Output: scan-report-path, vulnerabilities-found, gate-passed

**docker-build-push/action.yml:**
- Input: context, dockerfile, tags, platforms (default linux/amd64,linux/arm64), sign (default true)
- Steps: Setup Buildx, login to GHCR, build+push, Cosign sign, SBOM attach
- Output: image-digest, image-uri

**notify/action.yml:**
- Input: status (success/failure/warning), channel (default pipeline), title, body
- Steps: Format message, send to Slack webhook, optional Teams webhook
- Output: notification-id

**dora-record/action.yml:**
- Input: event-type (deploy/release/incident/recovery), environment, version
- Steps: Record timestamp, calculate metrics, write to GitHub artifact
- Output: deployment-frequency, lead-time, change-failure-rate

### TR-003: Environment Promotion Workflow

| Field | Value |
|-------|-------|
| **Priority** | P0 |
| **Gap** | G-003 |

**Specification:**
- Trigger: workflow_dispatch with source-env and target-env inputs
- Automatic: After successful dev deploy, auto-promote to staging (weekdays only)
- Manual gate: staging -> production requires manual approval via GitHub Environment
- Pre-promotion checks: All CI gates passed, security scan clean, no open critical CVEs
- Post-promotion: smoke tests, health checks, DORA metrics recording
- Rollback trigger: Automatic if smoke tests fail within 10 minutes

**Promotion Matrix:**
| Source | Target | Gate | Auto-promote |
|--------|--------|------|-------------|
| dev | staging | CI + Security pass | Yes (weekdays) |
| staging | production | Manual approval | No |

### TR-004: Rollback Workflow

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Gap** | G-004 |

**Specification:**
- Trigger: workflow_dispatch with environment and target-revision inputs
- Steps: Identify current deployment, kubectl rollout undo OR helm rollback, health check, notification
- Supports: Last known good (default), specific revision number, specific image tag
- Safety: Require environment protection rule approval for production
- Post-rollback: Smoke tests, incident recording for DORA metrics

### TR-005: Helm Chart CI

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Gap** | G-005 |

**Specification:**
- Trigger: PR changes to `deployment/helm/**` or `deployment/infrastructure/helm/**`
- Jobs: helm lint, helm template (dry-run), kubeval/kubeconform validation, chart-testing (ct lint-and-install)
- Matrix: All charts in repository
- Dependencies: kind cluster for chart-testing install step
- Outputs: Lint report, template diff

### TR-006: DORA Metrics Tracking

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Gap** | G-006 |

**Specification:**

**Four Key Metrics:**
1. **Deployment Frequency (DF)**: Tracked per environment via deploy workflow calls
2. **Lead Time for Changes (LT)**: Time from first commit to production deploy
3. **Mean Time to Recovery (MTTR)**: Time from incident to recovery (rollback/fix)
4. **Change Failure Rate (CFR)**: Failed deploys / total deploys

**Implementation:**
- `dora-record` composite action called from deploy, rollback, and release workflows
- Weekly aggregation workflow writes metrics to GitHub artifact + optional S3 bucket
- Grafana dashboard reads from Prometheus metrics (push via pushgateway) or JSON artifacts
- Target thresholds: DF >= 1/day, LT <= 1 day, MTTR <= 1 hour, CFR <= 15%

### TR-007: Pipeline Observability Dashboard

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Gap** | G-007 |

**Specification:**

**Grafana Dashboard Panels (12 panels):**

| Row | Panel | Metric |
|-----|-------|--------|
| DORA Metrics | Deployment Frequency | `cicd_deployments_total` by env |
| DORA Metrics | Lead Time for Changes | `cicd_lead_time_seconds` |
| DORA Metrics | Mean Time to Recovery | `cicd_mttr_seconds` |
| DORA Metrics | Change Failure Rate | `cicd_deploy_failures_total / cicd_deployments_total` |
| Pipeline Health | Workflow Success Rate | `github_workflow_runs{status="success"}` |
| Pipeline Health | Workflow Duration P95 | `github_workflow_duration_seconds` |
| Pipeline Health | Active Workflow Runs | `github_workflow_runs{status="in_progress"}` |
| Pipeline Health | Failed Workflows (24h) | `github_workflow_runs{status="failure"}` |
| Build Metrics | Build Duration Trend | `cicd_build_duration_seconds` |
| Build Metrics | Test Coverage Trend | `cicd_test_coverage_percent` |
| Security | Open CVEs by Severity | `cicd_cves_open{severity}` |
| Security | Security Scan Pass Rate | `cicd_security_scans{result="pass"}` |

### TR-008: Dependency Review

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Gap** | G-008 |

**Specification:**
- Use `actions/dependency-review-action@v4` on all PRs
- Block PRs introducing dependencies with known critical/high CVEs
- License allowlist: MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, PSF-2.0
- License denylist: GPL-2.0, GPL-3.0, AGPL-3.0 (copyleft)
- Output: Dependency diff comment on PR

### TR-009: Terraform Plan PR Comment

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Gap** | G-009 |

**Specification:**
- Trigger: PR changes to `deployment/terraform/**`
- Run `terraform plan` for each changed environment
- Post plan output as collapsible PR comment
- Include cost estimation (Infracost)
- Require manual approval before apply

### TR-010: Pipeline Alerts

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Gap** | G-007 (subset) |

**Specification:**

| Alert | Condition | Severity |
|-------|-----------|----------|
| PipelineFailureRate | >20% failure rate (1h window) | Critical |
| DeploymentStuck | Deployment pending >15m | Warning |
| SecurityScanFailed | Any security scan failure | Critical |
| CoverageDropped | Coverage below 85% | Warning |
| DORACFRHigh | Change failure rate >15% (7d) | Warning |
| DORALeadTimeHigh | Lead time >24h (7d avg) | Warning |
| HelmLintFailed | Helm chart validation failure | Warning |
| DependencyVulnerability | New critical CVE in deps | Critical |

---

## 5. Implementation Phases

### Phase 1: Foundation (P0 - Immediate)
- [x] Audit existing 15 workflows
- [x] Create CODEOWNERS file
- [x] Build 6 reusable composite actions
- [x] Build environment promotion workflow
- [x] Create Ralphy task tracking

### Phase 2: Resilience (P1)
- [x] Build rollback workflow
- [x] Build Helm chart CI workflow
- [x] Build DORA metrics tracking + aggregation
- [x] Build CI/CD Grafana dashboard
- [x] Build pipeline alerts (PrometheusRule)
- [x] Build dependency review workflow

### Phase 3: Enhancement (P2-P3)
- [x] Build Terraform plan PR comment workflow
- [ ] Configure ArgoCD sync trigger
- [ ] Add CI status badges to README
- [ ] Performance regression detection gates
- [ ] Unified notification strategy

---

## 6. Acceptance Criteria

| # | Criterion | Validation |
|---|-----------|------------|
| AC-001 | All 15 existing workflows remain functional | `gh workflow list` shows all active |
| AC-002 | CODEOWNERS routes reviews correctly | PR auto-assigns reviewers |
| AC-003 | Composite actions reduce duplication by 40%+ | Line count comparison |
| AC-004 | Environment promotion auto-triggers dev->staging | Verify via workflow runs |
| AC-005 | Rollback completes within 5 minutes | Timed rollback test |
| AC-006 | Helm charts validated on every PR | `helm-ci.yml` passes |
| AC-007 | DORA metrics dashboard shows all 4 metrics | Grafana panel verification |
| AC-008 | Pipeline alerts fire on threshold breach | Alert rule testing |
| AC-009 | Dependency review blocks known CVEs | Test with vulnerable dep |
| AC-010 | TF plan comments appear on infra PRs | PR comment verification |

---

## 7. Ralphy Task Checklist

- [x] INFRA-007-T01: Create CODEOWNERS file with team mappings
- [x] INFRA-007-T02: Build setup-greenlang composite action
- [x] INFRA-007-T03: Build setup-k8s composite action
- [x] INFRA-007-T04: Build security-gate composite action
- [x] INFRA-007-T05: Build docker-build-push composite action
- [x] INFRA-007-T06: Build notify composite action
- [x] INFRA-007-T07: Build dora-record composite action
- [x] INFRA-007-T08: Build promote-environment.yml workflow
- [x] INFRA-007-T09: Build rollback.yml workflow
- [x] INFRA-007-T10: Build helm-ci.yml workflow
- [x] INFRA-007-T11: Build dependency-review.yml workflow
- [x] INFRA-007-T12: Build dora-metrics.yml aggregation workflow
- [x] INFRA-007-T13: Build terraform-plan-comment.yml workflow
- [x] INFRA-007-T14: Build CI/CD Grafana dashboard (cicd-pipeline.json)
- [x] INFRA-007-T15: Build pipeline alerts (cicd-alerts.yaml)
- [ ] INFRA-007-T16: Validate all existing workflows still pass
- [ ] INFRA-007-T17: Update existing workflows to use composite actions
- [ ] INFRA-007-T18: End-to-end promotion test (dev -> staging)
- [ ] INFRA-007-T19: Rollback test on dev environment
- [ ] INFRA-007-T20: DORA metrics dashboard verification

---

## 8. Dependencies

| Dependency | Component | Required For |
|------------|-----------|--------------|
| INFRA-001 | K8s/EKS | Deployment targets, kubectl access |
| INFRA-002 | PostgreSQL | Database migration workflows |
| INFRA-003 | Redis | Cache invalidation on deploy |
| INFRA-004 | S3 | Artifact storage, DORA metrics persistence |
| INFRA-006 | Kong Gateway | API gateway config deployment |
| GitHub Environments | GitHub | Environment protection rules, secrets |
| Prometheus Pushgateway | Monitoring | DORA metrics ingestion |
| Slack Webhook | External | Pipeline notifications |

---

## 9. Security Considerations

- All secrets stored in GitHub Environments (never in workflow files)
- OIDC-based authentication for AWS (no long-lived credentials)
- Cosign signing for all container images
- SBOM generation for every release
- Dependency review blocks known CVEs in PRs
- Branch protection requires CODEOWNERS approval
- Production deploys require manual approval gate
- Audit trail via GitHub Actions workflow run logs (90-day retention)

---

## 10. Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Composite action breaking change | All workflows fail | Pin composite action versions, test in CI |
| DORA metrics data loss | No visibility into pipeline health | Dual-write to S3 + Prometheus |
| Auto-promotion false positive | Bad code in staging | Comprehensive smoke test suite |
| GitHub Actions rate limiting | Blocked deployments | Concurrency limits, self-hosted runners fallback |
| Secret rotation during deploy | Failed deployment | Use OIDC tokens, short-lived credentials |
