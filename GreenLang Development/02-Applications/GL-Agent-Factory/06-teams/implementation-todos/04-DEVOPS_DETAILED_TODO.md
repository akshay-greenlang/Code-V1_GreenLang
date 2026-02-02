# DevOps & Production Infrastructure - Detailed Implementation TODO

**Version:** 1.0.0
**Date:** 2025-12-04
**Owner:** GL-DevOpsEngineer
**Priority:** P1 HIGH PRIORITY
**Total Tasks:** 365
**Timeline:** 12 Weeks

---

## Executive Summary

This document provides the complete DevOps implementation roadmap for GreenLang Agent Factory production deployment. It covers CI/CD pipelines, Infrastructure as Code, Kubernetes operations, observability, and disaster recovery.

**Current State:** Development environment with local Kubernetes deployment
**Target State:** Multi-region production with 99.99% uptime, GitOps, full observability

---

# SECTION 1: CI/CD PIPELINE (85 Tasks)

## 1.1 GitHub Actions - PR Validation Workflow (12 Tasks)

### 1.1.1 Workflow File Creation
- [ ] **CICD-001** Create `.github/workflows/pr-validation-complete.yml` workflow file
  - **File:** `.github/workflows/pr-validation-complete.yml`
  - **Trigger:** Pull requests to `main`, `develop`, `release/*`
  - **Acceptance:** Workflow syntax valid

- [ ] **CICD-002** Configure Python version matrix (3.10, 3.11, 3.12)
  - **Configuration:** `strategy.matrix.python-version`
  - **Acceptance:** All Python versions tested in parallel

- [ ] **CICD-003** Add code checkout step with full history
  - **Action:** `actions/checkout@v4`
  - **Configuration:** `fetch-depth: 0`
  - **Acceptance:** Full git history available for analysis

- [ ] **CICD-004** Configure dependency caching with `actions/cache@v4`
  - **Cache Key:** `${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}`
  - **Paths:** `~/.cache/pip`, `~/.local`
  - **Acceptance:** Cache hit ratio > 80%

### 1.1.2 Code Quality Jobs
- [ ] **CICD-005** Add Python linting job (ruff, black, isort)
  - **Tools:** `ruff check .`, `black --check .`, `isort --check-only .`
  - **Timeout:** 5 minutes
  - **Acceptance:** Zero linting errors on main branch

- [ ] **CICD-006** Add type checking job with mypy strict mode
  - **Command:** `mypy --strict src/`
  - **Timeout:** 10 minutes
  - **Acceptance:** Zero type errors

- [ ] **CICD-007** Add unit test job with pytest (85% coverage gate)
  - **Command:** `pytest --cov=. --cov-fail-under=85 --cov-report=xml`
  - **Timeout:** 15 minutes
  - **Acceptance:** Coverage >= 85%

- [ ] **CICD-008** Add integration test job with service containers
  - **Services:** PostgreSQL 15, Redis 7
  - **Timeout:** 20 minutes
  - **Acceptance:** All integration tests pass

### 1.1.3 Infrastructure Validation Jobs
- [ ] **CICD-009** Add YAML lint job for Kubernetes manifests
  - **Tool:** `yamllint -d "{extends: default, rules: {line-length: disable}}"`
  - **Paths:** `kubernetes/`, `k8s/`, `infrastructure/`
  - **Acceptance:** Valid YAML syntax

- [ ] **CICD-010** Add Helm lint job for chart validation
  - **Command:** `helm lint infrastructure/helm/greenlang`
  - **Timeout:** 5 minutes
  - **Acceptance:** Valid Helm charts

- [ ] **CICD-011** Add Terraform validate job
  - **Commands:** `terraform init -backend=false`, `terraform validate`
  - **Paths:** `terraform/modules/*`
  - **Acceptance:** Valid Terraform configuration

- [ ] **CICD-012** Add Terraform fmt check job
  - **Command:** `terraform fmt -check -recursive`
  - **Acceptance:** Consistent formatting

---

## 1.2 GitHub Actions - Docker Build Workflow (15 Tasks)

### 1.2.1 Build Configuration
- [ ] **CICD-013** Create `.github/workflows/docker-build-production.yml`
  - **Triggers:** Push to main, release/*, tags v*
  - **Acceptance:** Workflow triggers correctly

- [ ] **CICD-014** Add Docker Buildx setup step
  - **Action:** `docker/setup-buildx-action@v3`
  - **Acceptance:** Buildx available for multi-arch builds

- [ ] **CICD-015** Add QEMU setup for multi-architecture builds
  - **Action:** `docker/setup-qemu-action@v3`
  - **Platforms:** `linux/amd64,linux/arm64`
  - **Acceptance:** ARM64 builds succeed

- [ ] **CICD-016** Add GHCR login step with GITHUB_TOKEN
  - **Action:** `docker/login-action@v3`
  - **Registry:** `ghcr.io`
  - **Acceptance:** Authenticated push to GHCR

- [ ] **CICD-017** Add AWS ECR login step (production registry)
  - **Action:** `aws-actions/amazon-ecr-login@v2`
  - **Region:** `us-east-1`
  - **Acceptance:** Authenticated push to ECR

### 1.2.2 Image Build Jobs
- [ ] **CICD-018** Configure build matrix for all agent images
  - **Matrix:** fuel-analyzer, carbon-intensity, energy-performance, eudr-compliance
  - **Parallel:** 4 concurrent builds
  - **Acceptance:** All images build successfully

- [ ] **CICD-019** Add base image build job
  - **Image:** `greenlang/base:python3.11`
  - **Size Target:** < 250MB
  - **Acceptance:** Base image available

- [ ] **CICD-020** Configure multi-stage build caching
  - **Cache From:** `type=gha`
  - **Cache To:** `type=gha,mode=max`
  - **Acceptance:** Cached builds < 2 minutes

- [ ] **CICD-021** Add image metadata extraction (labels, tags)
  - **Action:** `docker/metadata-action@v5`
  - **Tags:** Semantic version, SHA, latest
  - **Acceptance:** Proper OCI labels

- [ ] **CICD-022** Configure semantic versioning tags
  - **Pattern:** `v{{major}}.{{minor}}.{{patch}}`
  - **Acceptance:** Version tags applied correctly

### 1.2.3 Security and Distribution
- [ ] **CICD-023** Add Trivy vulnerability scan post-build
  - **Action:** `aquasecurity/trivy-action@master`
  - **Severity:** `CRITICAL,HIGH`
  - **Gate:** Fail on any CRITICAL CVE
  - **Acceptance:** Zero CRITICAL vulnerabilities

- [ ] **CICD-024** Add Cosign image signing step
  - **Tool:** `sigstore/cosign`
  - **Key:** AWS KMS key
  - **Acceptance:** Signed images verifiable

- [ ] **CICD-025** Add SBOM generation with Syft
  - **Format:** SPDX JSON
  - **Artifact:** Uploaded to release
  - **Acceptance:** SBOM attached to images

- [ ] **CICD-026** Push images to GHCR
  - **Registry:** `ghcr.io/greenlang`
  - **Visibility:** Private
  - **Acceptance:** Images available in GHCR

- [ ] **CICD-027** Push images to ECR (production)
  - **Registry:** `<account>.dkr.ecr.us-east-1.amazonaws.com`
  - **Acceptance:** Images available in ECR

---

## 1.3 GitHub Actions - Deployment Workflow (18 Tasks)

### 1.3.1 Staging Deployment
- [ ] **CICD-028** Create `.github/workflows/deploy-staging.yml`
  - **Trigger:** Push to main, workflow_dispatch
  - **Acceptance:** Manual trigger available

- [ ] **CICD-029** Add kubectl configuration step
  - **Action:** `azure/k8s-set-context@v3` or kubeconfig secret
  - **Cluster:** staging-eks-cluster
  - **Acceptance:** kubectl authenticated

- [ ] **CICD-030** Add Helm upgrade/install step
  - **Command:** `helm upgrade --install greenlang-agents ./infrastructure/helm/greenlang`
  - **Flags:** `--atomic --wait --timeout 10m`
  - **Values:** `values-staging.yaml`
  - **Acceptance:** Helm release deployed

- [ ] **CICD-031** Add deployment status verification
  - **Command:** `kubectl rollout status deployment -n greenlang-agents`
  - **Timeout:** 10 minutes
  - **Acceptance:** All pods ready

- [ ] **CICD-032** Run smoke tests post-deploy
  - **Script:** `./scripts/smoke-test.sh staging`
  - **Tests:** Health endpoints, basic API calls
  - **Acceptance:** Smoke tests pass

- [ ] **CICD-033** Add Slack notification (success/failure)
  - **Action:** `slackapi/slack-github-action@v1`
  - **Channel:** `#deployments`
  - **Acceptance:** Notifications sent

### 1.3.2 Production Deployment
- [ ] **CICD-034** Create `.github/workflows/deploy-production.yml`
  - **Trigger:** Manual approval required (GitHub Environments)
  - **Prerequisite:** Staging deployment successful
  - **Acceptance:** Approval gate enforced

- [ ] **CICD-035** Configure GitHub Environment protection rules
  - **Environment:** `production`
  - **Required Reviewers:** 2
  - **Wait Timer:** 5 minutes
  - **Acceptance:** Protection rules active

- [ ] **CICD-036** Implement blue-green deployment logic
  - **Strategy:** Deploy to green, run tests, switch traffic
  - **Script:** `./scripts/blue-green-deploy.sh`
  - **Acceptance:** Zero downtime deployment

- [ ] **CICD-037** Add canary deployment option (10% traffic)
  - **Tool:** Argo Rollouts or Flagger
  - **Metric:** Error rate < 1%
  - **Acceptance:** Canary gates functional

- [ ] **CICD-038** Configure automatic rollback on error rate
  - **Threshold:** Error rate > 1% for 5 minutes
  - **Action:** Automatic rollback to previous version
  - **Acceptance:** Rollback triggers automatically

- [ ] **CICD-039** Add PagerDuty notification for production
  - **Integration:** PagerDuty Events API v2
  - **Severity:** Critical for failures
  - **Acceptance:** Alerts sent to on-call

- [ ] **CICD-040** Add deployment audit log entry
  - **Storage:** S3 bucket or database
  - **Fields:** Version, deployer, timestamp, status
  - **Acceptance:** Audit trail complete

### 1.3.3 Rollback Workflow
- [ ] **CICD-041** Create `.github/workflows/rollback.yml`
  - **Trigger:** workflow_dispatch with inputs
  - **Inputs:** environment, revision
  - **Acceptance:** Manual rollback available

- [ ] **CICD-042** Add Helm release history query
  - **Command:** `helm history greenlang-agents -n greenlang-agents`
  - **Output:** Available revisions displayed
  - **Acceptance:** History visible

- [ ] **CICD-043** Execute Helm rollback command
  - **Command:** `helm rollback greenlang-agents <revision>`
  - **Acceptance:** Rollback executes successfully

- [ ] **CICD-044** Run post-rollback health checks
  - **Script:** `./scripts/health-check.sh`
  - **Acceptance:** Services healthy after rollback

- [ ] **CICD-045** Create automatic incident ticket on rollback
  - **Integration:** Jira, Linear, or GitHub Issues
  - **Template:** Rollback incident template
  - **Acceptance:** Ticket created with context

---

## 1.4 Security Scanning Workflow (15 Tasks)

### 1.4.1 Container Security
- [ ] **CICD-046** Create `.github/workflows/security-container-scan.yml`
  - **Trigger:** Push to main, pull_request, schedule (weekly)
  - **Acceptance:** Workflow runs on schedule

- [ ] **CICD-047** Add Trivy container vulnerability scan
  - **Severity:** CRITICAL, HIGH
  - **Output:** SARIF format
  - **Acceptance:** Results in GitHub Security tab

- [ ] **CICD-048** Add Snyk container security scan
  - **Integration:** Snyk GitHub Action
  - **Acceptance:** Snyk results visible

- [ ] **CICD-049** Configure vulnerability remediation SLA
  - **CRITICAL:** 24 hours
  - **HIGH:** 7 days
  - **Acceptance:** SLA documented and tracked

- [ ] **CICD-050** Add scan results to PR comment
  - **Action:** Custom comment with vulnerability summary
  - **Acceptance:** PR has security summary

### 1.4.2 Dependency Security
- [ ] **CICD-051** Create `.github/workflows/security-deps.yml`
  - **Trigger:** Pull request, weekly schedule
  - **Acceptance:** Dependency scanning active

- [ ] **CICD-052** Configure Safety scan for Python packages
  - **Command:** `safety check --json`
  - **Acceptance:** No known vulnerabilities in dependencies

- [ ] **CICD-053** Add npm audit for JavaScript dependencies
  - **Command:** `npm audit --audit-level=high`
  - **Acceptance:** No high-severity npm vulnerabilities

- [ ] **CICD-054** Configure license compliance scanning
  - **Tool:** `licensecheck` or `pip-licenses`
  - **Blocked:** GPL licenses in production code
  - **Acceptance:** License compliance report generated

- [ ] **CICD-055** Generate and publish SBOM
  - **Format:** SPDX or CycloneDX
  - **Artifact:** Attached to releases
  - **Acceptance:** SBOM available for each release

### 1.4.3 Infrastructure Security
- [ ] **CICD-056** Create `.github/workflows/security-iac.yml`
  - **Trigger:** Pull request affecting Terraform/Kubernetes
  - **Acceptance:** IaC security scanning active

- [ ] **CICD-057** Add tfsec for Terraform scanning
  - **Command:** `tfsec terraform/`
  - **Severity:** HIGH, CRITICAL
  - **Acceptance:** No HIGH severity issues

- [ ] **CICD-058** Add checkov for multi-IaC scanning
  - **Command:** `checkov -d terraform/ -d kubernetes/`
  - **Framework:** CIS benchmarks
  - **Acceptance:** CIS compliance verified

- [ ] **CICD-059** Add kube-linter for Kubernetes manifests
  - **Command:** `kube-linter lint kubernetes/`
  - **Acceptance:** Best practices enforced

- [ ] **CICD-060** Add kubesec for Kubernetes security scoring
  - **Command:** `kubesec scan kubernetes/*.yaml`
  - **Minimum Score:** 5
  - **Acceptance:** Security score meets threshold

---

## 1.5 Build Automation (10 Tasks)

- [ ] **CICD-061** Create Makefile for common build tasks
  - **File:** `Makefile.enhanced`
  - **Targets:** build, test, lint, docker-build, deploy
  - **Acceptance:** Make targets documented

- [ ] **CICD-062** Configure pre-commit hooks for build validation
  - **File:** `.pre-commit-config.yaml`
  - **Hooks:** black, ruff, mypy, yamllint
  - **Acceptance:** Pre-commit runs on commit

- [ ] **CICD-063** Create build artifact versioning scheme
  - **Format:** `v{major}.{minor}.{patch}-{sha}`
  - **Acceptance:** Version traceable to commit

- [ ] **CICD-064** Configure build notifications
  - **Channels:** Slack, email
  - **Events:** Build start, success, failure
  - **Acceptance:** Notifications sent

- [ ] **CICD-065** Create build metrics collection
  - **Metrics:** Build time, success rate, cache hits
  - **Storage:** Prometheus/CloudWatch
  - **Acceptance:** Build metrics visible in dashboard

- [ ] **CICD-066** Configure parallel build optimization
  - **Strategy:** Matrix builds for agents
  - **Concurrency:** 4 parallel jobs
  - **Acceptance:** Build time reduced 60%

- [ ] **CICD-067** Create build cache warming job
  - **Schedule:** Daily at 6 AM
  - **Purpose:** Pre-warm dependency caches
  - **Acceptance:** First build of day faster

- [ ] **CICD-068** Configure build artifact retention
  - **Docker Images:** 30 days for non-release
  - **Release Images:** Indefinite
  - **Acceptance:** Storage costs optimized

- [ ] **CICD-069** Create build failure analysis automation
  - **Tool:** GitHub Actions failure annotation
  - **Acceptance:** Clear failure messages

- [ ] **CICD-070** Document build process in runbook
  - **File:** `docs/runbooks/build-process.md`
  - **Acceptance:** Build process documented

---

## 1.6 Test Automation Integration (10 Tasks)

- [ ] **CICD-071** Create test database provisioning for CI
  - **Service:** PostgreSQL container
  - **Migrations:** Auto-apply in test
  - **Acceptance:** Fresh database per test run

- [ ] **CICD-072** Configure test data seeding
  - **Script:** `./scripts/seed-test-data.sh`
  - **Data:** Fixtures for each agent
  - **Acceptance:** Consistent test data

- [ ] **CICD-073** Create parallel test execution with pytest-xdist
  - **Workers:** `-n auto` (CPU-based)
  - **Acceptance:** Test time reduced 50%

- [ ] **CICD-074** Configure test result collection
  - **Format:** JUnit XML, Coverage XML
  - **Upload:** Codecov, SonarQube
  - **Acceptance:** Test results aggregated

- [ ] **CICD-075** Add flaky test detection and retry
  - **Plugin:** `pytest-rerunfailures`
  - **Max Reruns:** 2
  - **Acceptance:** Flaky tests identified

- [ ] **CICD-076** Create test timing reports
  - **Tool:** `pytest-durations`
  - **Threshold:** Flag tests > 10 seconds
  - **Acceptance:** Slow tests identified

- [ ] **CICD-077** Configure E2E test environment
  - **Namespace:** Ephemeral per test run
  - **Cleanup:** Automatic after tests
  - **Acceptance:** E2E tests isolated

- [ ] **CICD-078** Add test coverage trend tracking
  - **Tool:** Codecov badges
  - **Target:** Maintain 85%+
  - **Acceptance:** Coverage trend visible

- [ ] **CICD-079** Create performance regression tests
  - **Tool:** `pytest-benchmark`
  - **Baseline:** Stored in repo
  - **Acceptance:** Performance regression detected

- [ ] **CICD-080** Document test strategy
  - **File:** `docs/testing-strategy.md`
  - **Acceptance:** Test strategy documented

---

## 1.7 Rollback Procedures (5 Tasks)

- [ ] **CICD-081** Create automated rollback detection
  - **Metrics:** Error rate > 5%, latency > 2s
  - **Action:** Auto-rollback trigger
  - **Acceptance:** Rollback triggers automatically

- [ ] **CICD-082** Document rollback decision tree
  - **File:** `docs/runbooks/rollback-decision.md`
  - **Scenarios:** Covered for each failure type
  - **Acceptance:** Decision tree documented

- [ ] **CICD-083** Create rollback communication template
  - **Template:** Slack, email notification
  - **Stakeholders:** Engineering, Product, Support
  - **Acceptance:** Template created

- [ ] **CICD-084** Test rollback procedure monthly
  - **Schedule:** First Monday of month
  - **Documentation:** Results logged
  - **Acceptance:** Rollback tested regularly

- [ ] **CICD-085** Create rollback success verification
  - **Checks:** Health endpoints, key metrics
  - **Acceptance:** Rollback verified healthy

---

# SECTION 2: INFRASTRUCTURE AS CODE (95 Tasks)

## 2.1 Terraform Modules - VPC Module (8 Tasks)

- [ ] **IAC-001** Create `terraform/modules/vpc/main.tf`
  - **CIDR:** `10.0.0.0/16` (configurable)
  - **Acceptance:** VPC resource defined

- [ ] **IAC-002** Configure 3 public subnets across AZs
  - **CIDRs:** `10.0.1.0/24`, `10.0.2.0/24`, `10.0.3.0/24`
  - **Acceptance:** Public subnets in 3 AZs

- [ ] **IAC-003** Configure 3 private application subnets
  - **CIDRs:** `10.0.11.0/24`, `10.0.12.0/24`, `10.0.13.0/24`
  - **Acceptance:** Private app subnets created

- [ ] **IAC-004** Configure 3 private database subnets
  - **CIDRs:** `10.0.21.0/24`, `10.0.22.0/24`, `10.0.23.0/24`
  - **Acceptance:** Database subnets isolated

- [ ] **IAC-005** Configure NAT Gateways (3 for HA)
  - **Distribution:** 1 per AZ
  - **Acceptance:** HA NAT configuration

- [ ] **IAC-006** Add VPC Flow Logs to CloudWatch
  - **Traffic Type:** ALL
  - **Retention:** 30 days
  - **Acceptance:** Flow logs enabled

- [ ] **IAC-007** Configure VPC Endpoints (S3, ECR, SSM)
  - **Type:** Interface and Gateway
  - **Acceptance:** Private connectivity

- [ ] **IAC-008** Create `terraform/modules/vpc/outputs.tf`
  - **Outputs:** vpc_id, subnet_ids, route_table_ids
  - **Acceptance:** Outputs available for other modules

---

## 2.2 Terraform Modules - EKS Module (10 Tasks)

- [ ] **IAC-009** Create `terraform/modules/eks/main.tf`
  - **Version:** 1.28 (configurable)
  - **Acceptance:** EKS cluster resource defined

- [ ] **IAC-010** Enable OIDC provider for IRSA
  - **Purpose:** Pod IAM roles via service accounts
  - **Acceptance:** OIDC provider created

- [ ] **IAC-011** Configure cluster logging to CloudWatch
  - **Log Types:** api, audit, authenticator, controllerManager, scheduler
  - **Acceptance:** Control plane logs enabled

- [ ] **IAC-012** Create system node group (m6i.xlarge)
  - **Capacity:** Min 2, Max 4, Desired 2
  - **Labels:** `node-type: system`
  - **Acceptance:** System nodes ready

- [ ] **IAC-013** Create API gateway node group (c6i.2xlarge)
  - **Capacity:** Min 2, Max 8, Desired 3
  - **Labels:** `node-type: api`
  - **Acceptance:** API nodes ready

- [ ] **IAC-014** Create agent runtime node group (c6i.xlarge)
  - **Capacity:** Min 3, Max 20, Desired 5
  - **Labels:** `node-type: agent-runtime`
  - **Acceptance:** Agent nodes ready

- [ ] **IAC-015** Configure spot instances for worker nodes
  - **Spot Allocation:** diversified
  - **Fallback:** On-demand
  - **Acceptance:** Cost-optimized nodes

- [ ] **IAC-016** Configure cluster autoscaler IAM
  - **Permissions:** ASG scaling
  - **Acceptance:** Autoscaler IAM ready

- [ ] **IAC-017** Configure AWS Load Balancer Controller IAM
  - **Permissions:** ALB/NLB creation
  - **Acceptance:** LB controller IAM ready

- [ ] **IAC-018** Create `terraform/modules/eks/outputs.tf`
  - **Outputs:** cluster_endpoint, cluster_ca, oidc_provider_arn
  - **Acceptance:** EKS outputs available

---

## 2.3 Terraform Modules - RDS Module (8 Tasks)

- [ ] **IAC-019** Create `terraform/modules/rds/main.tf`
  - **Engine:** PostgreSQL 15
  - **Acceptance:** RDS resource defined

- [ ] **IAC-020** Configure Multi-AZ deployment
  - **Setting:** `multi_az = true`
  - **Acceptance:** HA database enabled

- [ ] **IAC-021** Configure storage (1TB io2 with IOPS)
  - **Storage Type:** io2
  - **IOPS:** 10000
  - **Acceptance:** High-performance storage

- [ ] **IAC-022** Enable storage encryption (AWS KMS)
  - **Key:** Customer-managed KMS key
  - **Acceptance:** Data encrypted at rest

- [ ] **IAC-023** Configure automated backups (30 days)
  - **Window:** 03:00-04:00 UTC
  - **Retention:** 30 days
  - **Acceptance:** Automated backups enabled

- [ ] **IAC-024** Enable Performance Insights
  - **Retention:** 7 days (free tier)
  - **Acceptance:** Performance data collected

- [ ] **IAC-025** Create read replicas (3 for prod)
  - **Regions:** Same region, different AZs
  - **Acceptance:** Read scaling available

- [ ] **IAC-026** Create `terraform/modules/rds/outputs.tf`
  - **Outputs:** endpoint, reader_endpoint, security_group_id
  - **Acceptance:** RDS outputs available

---

## 2.4 Terraform Modules - ElastiCache Module (6 Tasks)

- [ ] **IAC-027** Create `terraform/modules/elasticache/main.tf`
  - **Engine:** Redis 7.0
  - **Mode:** Cluster mode enabled
  - **Acceptance:** Redis cluster defined

- [ ] **IAC-028** Configure replication group (3 primaries + 3 replicas)
  - **Shards:** 3
  - **Replicas per Shard:** 1
  - **Acceptance:** HA Redis cluster

- [ ] **IAC-029** Enable Multi-AZ with automatic failover
  - **Setting:** `automatic_failover_enabled = true`
  - **Acceptance:** Automatic failover configured

- [ ] **IAC-030** Enable encryption at rest and in transit
  - **At Rest:** KMS key
  - **In Transit:** TLS
  - **Acceptance:** Encryption enabled

- [ ] **IAC-031** Configure snapshot settings
  - **Window:** 05:00-06:00 UTC
  - **Retention:** 7 days
  - **Acceptance:** Snapshots enabled

- [ ] **IAC-032** Create `terraform/modules/elasticache/outputs.tf`
  - **Outputs:** configuration_endpoint, security_group_id
  - **Acceptance:** Redis outputs available

---

## 2.5 Terraform Modules - S3/Storage Module (6 Tasks)

- [ ] **IAC-033** Create `terraform/modules/s3/main.tf`
  - **Buckets:** artifacts, audit-logs, backups, terraform-state
  - **Acceptance:** S3 buckets defined

- [ ] **IAC-034** Enable versioning on all buckets
  - **Setting:** `versioning.enabled = true`
  - **Acceptance:** Version history preserved

- [ ] **IAC-035** Configure lifecycle rules (IA transition)
  - **Transition:** 30 days to IA, 90 days to Glacier
  - **Acceptance:** Cost-optimized storage

- [ ] **IAC-036** Configure cross-region replication (DR)
  - **Destination:** DR region bucket
  - **Acceptance:** Data replicated to DR

- [ ] **IAC-037** Block all public access
  - **Settings:** All block public access options enabled
  - **Acceptance:** No public S3 access

- [ ] **IAC-038** Create `terraform/modules/s3/outputs.tf`
  - **Outputs:** bucket_arns, bucket_names
  - **Acceptance:** S3 outputs available

---

## 2.6 Terraform Modules - IAM Module (8 Tasks)

- [ ] **IAC-039** Create `terraform/modules/iam/main.tf`
  - **Purpose:** Centralized IAM role management
  - **Acceptance:** IAM module created

- [ ] **IAC-040** Create EKS cluster role
  - **Permissions:** AmazonEKSClusterPolicy
  - **Acceptance:** Cluster role created

- [ ] **IAC-041** Create EKS node role
  - **Permissions:** AmazonEKSWorkerNodePolicy, AmazonEKS_CNI_Policy, AmazonEC2ContainerRegistryReadOnly
  - **Acceptance:** Node role created

- [ ] **IAC-042** Create agent service account roles (IRSA)
  - **Agents:** fuel-analyzer, carbon-intensity, energy-performance, eudr-compliance
  - **Acceptance:** Per-agent IAM roles

- [ ] **IAC-043** Create CI/CD deployment role
  - **Permissions:** EKS, ECR, S3 access
  - **Trust:** GitHub OIDC provider
  - **Acceptance:** CI/CD can deploy

- [ ] **IAC-044** Create monitoring role
  - **Permissions:** CloudWatch, X-Ray
  - **Acceptance:** Monitoring data access

- [ ] **IAC-045** Create backup role
  - **Permissions:** S3, RDS snapshots
  - **Acceptance:** Backup automation enabled

- [ ] **IAC-046** Create `terraform/modules/iam/outputs.tf`
  - **Outputs:** role_arns, instance_profile_arns
  - **Acceptance:** IAM outputs available

---

## 2.7 Helm Charts (25 Tasks)

### 2.7.1 Umbrella Chart Structure
- [ ] **HELM-001** Create `helm/greenlang-agents/Chart.yaml`
  - **Version:** 1.0.0
  - **Dependencies:** Sub-charts for each agent
  - **Acceptance:** Chart metadata complete

- [ ] **HELM-002** Create `helm/greenlang-agents/values.yaml`
  - **Defaults:** Common configuration
  - **Acceptance:** Default values defined

- [ ] **HELM-003** Create `helm/greenlang-agents/values-dev.yaml`
  - **Replicas:** 1
  - **Resources:** Minimal
  - **Acceptance:** Dev values ready

- [ ] **HELM-004** Create `helm/greenlang-agents/values-staging.yaml`
  - **Replicas:** 2
  - **Resources:** Moderate
  - **Acceptance:** Staging values ready

- [ ] **HELM-005** Create `helm/greenlang-agents/values-prod.yaml`
  - **Replicas:** 3+
  - **Resources:** Production-grade
  - **Acceptance:** Prod values ready

### 2.7.2 Base Templates
- [ ] **HELM-006** Create `templates/_helpers.tpl`
  - **Helpers:** fullname, labels, selectorLabels
  - **Acceptance:** Template helpers working

- [ ] **HELM-007** Create ServiceAccount template
  - **Annotations:** IAM role ARN
  - **Acceptance:** SA created with IRSA

- [ ] **HELM-008** Create RBAC Role and RoleBinding templates
  - **Permissions:** ConfigMap read, Secret read
  - **Acceptance:** RBAC applied

- [ ] **HELM-009** Create ResourceQuota template
  - **Limits:** CPU, memory, pods
  - **Acceptance:** Quotas enforced

- [ ] **HELM-010** Create PodDisruptionBudget template
  - **MinAvailable:** 2 for production
  - **Acceptance:** PDB protects availability

- [ ] **HELM-011** Create NetworkPolicy templates
  - **Default:** Deny all ingress
  - **Allow:** Specific service communication
  - **Acceptance:** Network isolation

### 2.7.3 Agent Sub-Charts
- [ ] **HELM-012** Create `charts/fuel-analyzer/Chart.yaml`
  - **Version:** Match agent version
  - **Acceptance:** Sub-chart created

- [ ] **HELM-013** Create fuel-analyzer Deployment template
  - **Container:** Image, ports, env, resources
  - **Probes:** Liveness, readiness, startup
  - **Acceptance:** Deployment template complete

- [ ] **HELM-014** Create fuel-analyzer Service template
  - **Type:** ClusterIP
  - **Ports:** 8000
  - **Acceptance:** Service template complete

- [ ] **HELM-015** Create fuel-analyzer HPA template
  - **Min:** 2, Max: 10
  - **Metrics:** CPU 70%, Memory 80%
  - **Acceptance:** HPA template complete

- [ ] **HELM-016** Create fuel-analyzer ServiceMonitor template
  - **Endpoint:** /metrics
  - **Interval:** 30s
  - **Acceptance:** Prometheus scraping

- [ ] **HELM-017** Create carbon-intensity sub-chart
  - **Same Pattern:** As fuel-analyzer
  - **Acceptance:** Sub-chart complete

- [ ] **HELM-018** Create energy-performance sub-chart
  - **Same Pattern:** As fuel-analyzer
  - **Acceptance:** Sub-chart complete

- [ ] **HELM-019** Create eudr-compliance sub-chart (CRITICAL)
  - **Replicas:** Min 3 (stricter SLA)
  - **PDB:** MinAvailable 2
  - **Acceptance:** HA configuration

### 2.7.4 Ingress and TLS
- [ ] **HELM-020** Create Ingress template
  - **Annotations:** Rate limiting, CORS
  - **TLS:** cert-manager integration
  - **Acceptance:** Ingress with TLS

- [ ] **HELM-021** Create Certificate template (cert-manager)
  - **Issuer:** Let's Encrypt production
  - **Acceptance:** TLS certificate issued

- [ ] **HELM-022** Create ClusterIssuer for Let's Encrypt
  - **Type:** HTTP01 challenge
  - **Acceptance:** Certificate automation

### 2.7.5 Helmfile Configuration
- [ ] **HELM-023** Create `helmfile.yaml`
  - **Environments:** dev, staging, prod
  - **Releases:** All agent charts
  - **Acceptance:** Helmfile deployable

- [ ] **HELM-024** Configure Helm repository sources
  - **Repos:** Bitnami, Prometheus Community
  - **Acceptance:** Dependencies available

- [ ] **HELM-025** Add Helmfile hooks for validation
  - **Pre-Install:** Validate values
  - **Post-Install:** Run smoke tests
  - **Acceptance:** Hooks execute

---

## 2.8 Kustomize Overlays (10 Tasks)

- [ ] **KUST-001** Create `kustomize/base/kustomization.yaml`
  - **Resources:** Base manifests
  - **Acceptance:** Base configuration

- [ ] **KUST-002** Add common labels and annotations
  - **Labels:** app, version, environment
  - **Acceptance:** Consistent labeling

- [ ] **KUST-003** Create `kustomize/overlays/dev/kustomization.yaml`
  - **Patches:** Reduced replicas, resources
  - **Acceptance:** Dev overlay working

- [ ] **KUST-004** Configure dev-specific ConfigMaps
  - **Settings:** Debug logging, dev endpoints
  - **Acceptance:** Dev config applied

- [ ] **KUST-005** Create `kustomize/overlays/staging/kustomization.yaml`
  - **Patches:** Staging replicas, resources
  - **Acceptance:** Staging overlay working

- [ ] **KUST-006** Configure staging-specific ConfigMaps
  - **Settings:** Info logging, staging endpoints
  - **Acceptance:** Staging config applied

- [ ] **KUST-007** Create `kustomize/overlays/prod/kustomization.yaml`
  - **Patches:** Production replicas, resources
  - **Acceptance:** Prod overlay working

- [ ] **KUST-008** Configure production-specific ConfigMaps
  - **Settings:** Warn logging, production endpoints
  - **Acceptance:** Prod config applied

- [ ] **KUST-009** Add anti-affinity rules for production
  - **Topology:** Zone-based spread
  - **Acceptance:** Pods distributed across AZs

- [ ] **KUST-010** Create secret management with sealed-secrets
  - **Tool:** Bitnami Sealed Secrets
  - **Acceptance:** Secrets encrypted in Git

---

# SECTION 3: KUBERNETES OPERATIONS (75 Tasks)

## 3.1 Cluster Setup - Dev Environment (10 Tasks)

- [ ] **K8S-001** Apply Terraform for dev VPC
  - **Command:** `terraform apply -var-file=dev.tfvars`
  - **Acceptance:** Dev VPC created

- [ ] **K8S-002** Apply Terraform for dev EKS
  - **Command:** `terraform apply -target=module.eks`
  - **Acceptance:** Dev EKS cluster running

- [ ] **K8S-003** Configure kubectl for dev cluster
  - **Command:** `aws eks update-kubeconfig --name dev-greenlang`
  - **Acceptance:** kubectl authenticated

- [ ] **K8S-004** Install NGINX Ingress Controller
  - **Helm:** `helm install ingress-nginx ingress-nginx/ingress-nginx`
  - **Acceptance:** Ingress controller running

- [ ] **K8S-005** Install cert-manager
  - **Helm:** `helm install cert-manager jetstack/cert-manager`
  - **Acceptance:** Certificate management ready

- [ ] **K8S-006** Install External Secrets Operator
  - **Helm:** `helm install external-secrets external-secrets/external-secrets`
  - **Acceptance:** AWS Secrets Manager integration

- [ ] **K8S-007** Install metrics-server
  - **Helm:** `helm install metrics-server metrics-server/metrics-server`
  - **Acceptance:** kubectl top working

- [ ] **K8S-008** Install cluster-autoscaler
  - **Helm:** Configure for EKS
  - **Acceptance:** Auto-scaling nodes

- [ ] **K8S-009** Validate dev cluster health
  - **Command:** `kubectl get nodes`, `kubectl get pods -A`
  - **Acceptance:** All components healthy

- [ ] **K8S-010** Document dev cluster access
  - **File:** `docs/clusters/dev-access.md`
  - **Acceptance:** Access documented

## 3.2 Cluster Setup - Staging Environment (8 Tasks)

- [ ] **K8S-011** Apply Terraform for staging VPC
  - **Acceptance:** Staging VPC created

- [ ] **K8S-012** Apply Terraform for staging EKS
  - **Acceptance:** Staging EKS cluster running

- [ ] **K8S-013** Configure kubectl for staging cluster
  - **Acceptance:** kubectl authenticated

- [ ] **K8S-014** Install all Kubernetes add-ons (staging)
  - **Add-ons:** Ingress, cert-manager, external-secrets, metrics-server, autoscaler
  - **Acceptance:** All add-ons running

- [ ] **K8S-015** Configure HPA for staging
  - **Min:** 2, Max: 5
  - **Acceptance:** HPA configured

- [ ] **K8S-016** Deploy monitoring stack to staging
  - **Components:** Prometheus, Grafana, Loki
  - **Acceptance:** Monitoring ready

- [ ] **K8S-017** Validate staging cluster health
  - **Acceptance:** All components healthy

- [ ] **K8S-018** Document staging cluster access
  - **Acceptance:** Access documented

## 3.3 Cluster Setup - Production Environment (12 Tasks)

- [ ] **K8S-019** Apply Terraform for prod VPC (us-east-1)
  - **Acceptance:** Production VPC created

- [ ] **K8S-020** Apply Terraform for prod EKS
  - **Acceptance:** Production EKS cluster running

- [ ] **K8S-021** Configure production node groups
  - **Groups:** system, api, agent-runtime
  - **Acceptance:** Node groups ready

- [ ] **K8S-022** Install all Kubernetes add-ons (production)
  - **Acceptance:** All add-ons running

- [ ] **K8S-023** Configure production HPA settings
  - **Min:** 3, Max: 20
  - **Acceptance:** HPA configured

- [ ] **K8S-024** Enable Pod Security Admission (restricted)
  - **Mode:** Enforce
  - **Acceptance:** PSA enabled

- [ ] **K8S-025** Configure audit logging
  - **Destination:** CloudWatch Logs
  - **Acceptance:** Audit logs collected

- [ ] **K8S-026** Configure backup scheduling (Velero)
  - **Schedule:** Daily 2 AM UTC
  - **Retention:** 30 days
  - **Acceptance:** Backups automated

- [ ] **K8S-027** Deploy monitoring stack to production
  - **Acceptance:** Monitoring ready

- [ ] **K8S-028** Validate production cluster health
  - **Acceptance:** All components healthy

- [ ] **K8S-029** Document production access (restricted)
  - **Access:** Require VPN + MFA
  - **Acceptance:** Access documented

- [ ] **K8S-030** Create production cluster runbook
  - **File:** `docs/runbooks/production-cluster.md`
  - **Acceptance:** Runbook complete

## 3.4 Cluster Setup - DR Environment (8 Tasks)

- [ ] **K8S-031** Apply Terraform for DR VPC (us-west-2)
  - **Acceptance:** DR VPC created

- [ ] **K8S-032** Apply Terraform for DR EKS
  - **Nodes:** Scaled down (standby)
  - **Acceptance:** DR EKS cluster running

- [ ] **K8S-033** Configure cross-region networking
  - **VPC Peering:** Or Transit Gateway
  - **Acceptance:** Cross-region connectivity

- [ ] **K8S-034** Install all Kubernetes add-ons (DR)
  - **Acceptance:** All add-ons running

- [ ] **K8S-035** Configure DR database replica
  - **RDS:** Cross-region read replica
  - **Acceptance:** Data replicated

- [ ] **K8S-036** Configure S3 cross-region replication
  - **Acceptance:** Objects replicated to DR

- [ ] **K8S-037** Test failover procedures
  - **Test:** Simulate primary failure
  - **Acceptance:** Failover tested

- [ ] **K8S-038** Document DR cluster access
  - **Acceptance:** DR procedures documented

## 3.5 Namespace Design (10 Tasks)

- [ ] **K8S-039** Create greenlang-system namespace
  - **Purpose:** Core infrastructure components
  - **Acceptance:** Namespace created

- [ ] **K8S-040** Create greenlang-agents namespace
  - **Purpose:** Agent deployments
  - **Acceptance:** Namespace created

- [ ] **K8S-041** Create greenlang-monitoring namespace
  - **Purpose:** Prometheus, Grafana, alerting
  - **Acceptance:** Namespace created

- [ ] **K8S-042** Create greenlang-logging namespace
  - **Purpose:** EFK/Loki stack
  - **Acceptance:** Namespace created

- [ ] **K8S-043** Create greenlang-ingress namespace
  - **Purpose:** Ingress controllers
  - **Acceptance:** Namespace created

- [ ] **K8S-044** Apply namespace labels
  - **Labels:** environment, team, cost-center
  - **Acceptance:** Labels applied

- [ ] **K8S-045** Apply Pod Security labels (restricted)
  - **Label:** `pod-security.kubernetes.io/enforce: restricted`
  - **Acceptance:** PSA enforced

- [ ] **K8S-046** Create ResourceQuota for dev
  - **CPU:** 50 cores, **Memory:** 100GB
  - **Acceptance:** Quota enforced

- [ ] **K8S-047** Create ResourceQuota for staging
  - **CPU:** 75 cores, **Memory:** 150GB
  - **Acceptance:** Quota enforced

- [ ] **K8S-048** Create ResourceQuota for production
  - **CPU:** 200 cores, **Memory:** 400GB
  - **Acceptance:** Quota enforced

## 3.6 Resource Quotas and Limits (7 Tasks)

- [ ] **K8S-049** Create LimitRange for all namespaces
  - **Default CPU Request:** 100m
  - **Default Memory Request:** 128Mi
  - **Acceptance:** Limits applied

- [ ] **K8S-050** Set max CPU limit per container
  - **Max:** 4 cores
  - **Acceptance:** Limit enforced

- [ ] **K8S-051** Set max memory limit per container
  - **Max:** 8Gi
  - **Acceptance:** Limit enforced

- [ ] **K8S-052** Configure quota usage alerts
  - **Threshold:** 80% usage
  - **Acceptance:** Alerts configured

- [ ] **K8S-053** Document quota increase process
  - **File:** `docs/operations/quota-increase.md`
  - **Acceptance:** Process documented

- [ ] **K8S-054** Create resource request best practices guide
  - **File:** `docs/best-practices/resource-requests.md`
  - **Acceptance:** Guide created

- [ ] **K8S-055** Implement VPA in recommendation mode
  - **Purpose:** Right-sizing recommendations
  - **Acceptance:** VPA collecting data

## 3.7 Network Policies (10 Tasks)

- [ ] **K8S-056** Create default-deny-all NetworkPolicy
  - **Effect:** Block all ingress by default
  - **Acceptance:** Default deny applied

- [ ] **K8S-057** Create allow-dns NetworkPolicy
  - **Allow:** Port 53 to kube-dns
  - **Acceptance:** DNS resolution works

- [ ] **K8S-058** Create allow-ingress NetworkPolicy
  - **Allow:** From ingress-nginx namespace
  - **Acceptance:** External traffic works

- [ ] **K8S-059** Create allow-prometheus-scrape NetworkPolicy
  - **Allow:** From monitoring namespace on metrics port
  - **Acceptance:** Metrics collection works

- [ ] **K8S-060** Create inter-agent communication policy
  - **Allow:** Between agents in same namespace
  - **Acceptance:** Agent-to-agent calls work

- [ ] **K8S-061** Create database access policy
  - **Allow:** To PostgreSQL on port 5432
  - **Acceptance:** Database connectivity

- [ ] **K8S-062** Create Redis access policy
  - **Allow:** To Redis on port 6379
  - **Acceptance:** Cache connectivity

- [ ] **K8S-063** Create external API access policy
  - **Allow:** HTTPS (443) egress only
  - **Acceptance:** External APIs accessible

- [ ] **K8S-064** Test network isolation
  - **Tool:** Network policy tester
  - **Acceptance:** Isolation verified

- [ ] **K8S-065** Document network policies
  - **File:** `docs/security/network-policies.md`
  - **Acceptance:** Policies documented

## 3.8 Storage Classes (5 Tasks)

- [ ] **K8S-066** Create StorageClass for gp3 (default)
  - **Parameters:** gp3, 3000 IOPS baseline
  - **Acceptance:** Default storage class

- [ ] **K8S-067** Create StorageClass for io2 (database)
  - **Parameters:** io2, 10000 IOPS
  - **Acceptance:** High-IOPS storage class

- [ ] **K8S-068** Configure volume expansion
  - **AllowVolumeExpansion:** true
  - **Acceptance:** PVC expansion enabled

- [ ] **K8S-069** Set PV reclaim policy (Retain for prod)
  - **Policy:** Retain
  - **Acceptance:** Data preserved on delete

- [ ] **K8S-070** Document storage classes
  - **File:** `docs/infrastructure/storage-classes.md`
  - **Acceptance:** Storage documented

## 3.9 Ingress with TLS (5 Tasks)

- [ ] **K8S-071** Deploy NGINX Ingress Controller
  - **Helm Chart:** ingress-nginx
  - **Acceptance:** Controller running

- [ ] **K8S-072** Configure Ingress annotations
  - **Rate Limiting:** 1000 req/s
  - **SSL Redirect:** true
  - **CORS:** Configured
  - **Acceptance:** Annotations applied

- [ ] **K8S-073** Set up Let's Encrypt ClusterIssuer
  - **Type:** HTTP01 or DNS01 challenge
  - **Environment:** Production issuer
  - **Acceptance:** Certificates issued

- [ ] **K8S-074** Create wildcard certificate
  - **Domain:** *.greenlang.io
  - **Acceptance:** Wildcard cert issued

- [ ] **K8S-075** Configure DNS records (Route 53)
  - **Records:** api.greenlang.io, agents.greenlang.io
  - **Acceptance:** DNS resolves correctly

---

# SECTION 4: OBSERVABILITY STACK (70 Tasks)

## 4.1 ELK/EFK Logging (18 Tasks)

### 4.1.1 Elasticsearch Deployment
- [ ] **OBS-001** Deploy Elasticsearch (3 nodes)
  - **Helm:** elastic/elasticsearch
  - **Replicas:** 3
  - **Acceptance:** ES cluster green

- [ ] **OBS-002** Configure Elasticsearch persistence (500GB)
  - **Storage Class:** io2
  - **Acceptance:** Data persisted

- [ ] **OBS-003** Configure Elasticsearch replicas
  - **Index Replicas:** 1
  - **Acceptance:** Data replicated

- [ ] **OBS-004** Create Elasticsearch index templates
  - **Templates:** logs-*, metrics-*, traces-*
  - **Acceptance:** Index templates applied

- [ ] **OBS-005** Configure ILM policies (hot/warm/cold)
  - **Hot:** 7 days
  - **Warm:** 30 days
  - **Cold:** 365 days
  - **Acceptance:** ILM policies active

### 4.1.2 Kibana Deployment
- [ ] **OBS-006** Deploy Kibana
  - **Helm:** elastic/kibana
  - **Acceptance:** Kibana accessible

- [ ] **OBS-007** Configure Kibana authentication
  - **Method:** SAML/OIDC with Okta
  - **Acceptance:** SSO enabled

- [ ] **OBS-008** Create Kibana index patterns
  - **Patterns:** logs-*, application-*
  - **Acceptance:** Patterns created

### 4.1.3 Log Shipping
- [ ] **OBS-009** Deploy Fluent Bit DaemonSet
  - **Helm:** fluent/fluent-bit
  - **Acceptance:** Running on all nodes

- [ ] **OBS-010** Configure Fluent Bit parsers
  - **Parsers:** JSON, multiline Python
  - **Acceptance:** Logs parsed correctly

- [ ] **OBS-011** Configure Fluent Bit filters
  - **Filters:** Kubernetes metadata enrichment
  - **Acceptance:** Metadata added

- [ ] **OBS-012** Configure Elasticsearch output
  - **Index:** logs-{kubernetes.namespace}-*
  - **Acceptance:** Logs flowing to ES

- [ ] **OBS-013** Configure buffer settings
  - **Buffer:** 5MB flush interval 5s
  - **Acceptance:** Buffering configured

- [ ] **OBS-014** Set up dead letter queue
  - **DLQ:** S3 bucket for failed logs
  - **Acceptance:** DLQ configured

### 4.1.4 Kibana Dashboards
- [ ] **OBS-015** Create Agent Logs Overview dashboard
  - **Panels:** Log volume, error rate, top errors
  - **Acceptance:** Dashboard created

- [ ] **OBS-016** Create Error Logs dashboard
  - **Panels:** Error trends, stack traces, affected agents
  - **Acceptance:** Dashboard created

- [ ] **OBS-017** Create Request Trace dashboard
  - **Panels:** Request flow, latency distribution
  - **Acceptance:** Dashboard created

- [ ] **OBS-018** Configure Kibana alerts
  - **Alerts:** Error spike, log volume drop
  - **Acceptance:** Alerts configured

## 4.2 Prometheus Metrics (17 Tasks)

### 4.2.1 Prometheus Deployment
- [ ] **OBS-019** Deploy Prometheus Operator via Helm
  - **Helm:** prometheus-community/kube-prometheus-stack
  - **Acceptance:** Prometheus running

- [ ] **OBS-020** Configure Prometheus persistence (100GB)
  - **Retention:** 30 days
  - **Acceptance:** Metrics persisted

- [ ] **OBS-021** Configure service discovery
  - **Method:** Kubernetes SD
  - **Acceptance:** Targets discovered

- [ ] **OBS-022** Configure Alertmanager integration
  - **Receivers:** Slack, PagerDuty
  - **Acceptance:** Alerts routing

### 4.2.2 ServiceMonitor Configuration
- [ ] **OBS-023** Create ServiceMonitor for fuel-analyzer
  - **Endpoint:** /metrics, Interval: 30s
  - **Acceptance:** Metrics scraped

- [ ] **OBS-024** Create ServiceMonitor for carbon-intensity
  - **Acceptance:** Metrics scraped

- [ ] **OBS-025** Create ServiceMonitor for energy-performance
  - **Acceptance:** Metrics scraped

- [ ] **OBS-026** Create ServiceMonitor for eudr-compliance
  - **Acceptance:** Metrics scraped

- [ ] **OBS-027** Create ServiceMonitor for ingress-nginx
  - **Acceptance:** Ingress metrics scraped

- [ ] **OBS-028** Create ServiceMonitor for postgres-exporter
  - **Acceptance:** DB metrics scraped

- [ ] **OBS-029** Create ServiceMonitor for redis-exporter
  - **Acceptance:** Redis metrics scraped

### 4.2.3 Custom Metrics
- [ ] **OBS-030** Add agent_requests_total counter
  - **Labels:** agent, method, status
  - **Acceptance:** Request counting

- [ ] **OBS-031** Add agent_request_duration_seconds histogram
  - **Buckets:** 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
  - **Acceptance:** Latency tracking

- [ ] **OBS-032** Add agent_calculations_total counter
  - **Labels:** agent, tool_name, status
  - **Acceptance:** Calculation tracking

- [ ] **OBS-033** Add agent_cache_hits_total counter
  - **Labels:** agent, cache_type
  - **Acceptance:** Cache hit ratio

- [ ] **OBS-034** Add agent_llm_tokens_total counter
  - **Labels:** agent, model, type (input/output)
  - **Acceptance:** LLM usage tracking

- [ ] **OBS-035** Add agent_provenance_hashes_total counter
  - **Labels:** agent
  - **Acceptance:** Provenance tracking

## 4.3 Jaeger Distributed Tracing (12 Tasks)

### 4.3.1 OpenTelemetry Setup
- [ ] **OBS-036** Deploy OpenTelemetry Collector
  - **Helm:** open-telemetry/opentelemetry-collector
  - **Acceptance:** Collector running

- [ ] **OBS-037** Configure OTLP receiver
  - **Ports:** 4317 (gRPC), 4318 (HTTP)
  - **Acceptance:** OTLP receiving traces

- [ ] **OBS-038** Configure trace sampling (10% prod)
  - **Type:** Probabilistic
  - **Acceptance:** Sampling configured

- [ ] **OBS-039** Configure batch processor
  - **Size:** 512 spans
  - **Timeout:** 5s
  - **Acceptance:** Batch processing

- [ ] **OBS-040** Configure Jaeger exporter
  - **Endpoint:** jaeger-collector:14268
  - **Acceptance:** Traces in Jaeger

### 4.3.2 Jaeger Deployment
- [ ] **OBS-041** Deploy Jaeger All-in-One (dev)
  - **Helm:** jaegertracing/jaeger
  - **Acceptance:** Jaeger UI accessible

- [ ] **OBS-042** Deploy Jaeger with Elasticsearch (prod)
  - **Storage:** Elasticsearch
  - **Acceptance:** Persistent traces

- [ ] **OBS-043** Configure trace retention
  - **Retention:** 7 days
  - **Acceptance:** Old traces deleted

### 4.3.3 Application Instrumentation
- [ ] **OBS-044** Add OpenTelemetry SDK to agents
  - **Package:** opentelemetry-sdk
  - **Acceptance:** SDK initialized

- [ ] **OBS-045** Configure auto-instrumentation for FastAPI
  - **Package:** opentelemetry-instrumentation-fastapi
  - **Acceptance:** HTTP traces captured

- [ ] **OBS-046** Add custom spans for tool execution
  - **Spans:** Tool name, duration, status
  - **Acceptance:** Tool-level visibility

- [ ] **OBS-047** Configure trace context propagation
  - **Headers:** traceparent, tracestate
  - **Acceptance:** Distributed traces linked

## 4.4 Grafana Dashboards (15 Tasks)

### 4.4.1 Overview Dashboards
- [ ] **OBS-048** Create Agent Factory Overview dashboard
  - **Panels:** Total agents, request rate, error rate, latency
  - **Acceptance:** Overview visible

- [ ] **OBS-049** Create Agent Detail dashboard
  - **Variable:** Agent selector
  - **Panels:** Per-agent metrics
  - **Acceptance:** Agent drill-down

- [ ] **OBS-050** Create SLO Dashboard
  - **Panels:** Availability, latency, error budget
  - **Acceptance:** SLO tracking

- [ ] **OBS-051** Create Infrastructure Dashboard
  - **Panels:** Node health, pod distribution, storage
  - **Acceptance:** Infra visibility

- [ ] **OBS-052** Create Cost Dashboard
  - **Panels:** Compute cost, LLM cost, storage cost
  - **Acceptance:** Cost visibility

### 4.4.2 Alert Dashboards
- [ ] **OBS-053** Create Alert Overview dashboard
  - **Panels:** Active alerts, alert history, alert trends
  - **Acceptance:** Alert visibility

- [ ] **OBS-054** Create On-Call Dashboard
  - **Panels:** Current on-call, response times, escalations
  - **Acceptance:** On-call visibility

### 4.4.3 PrometheusRule Configuration
- [ ] **OBS-055** Create alert: AgentHighErrorRate
  - **Condition:** Error rate > 1% for 5 minutes
  - **Severity:** Critical
  - **Acceptance:** Alert fires

- [ ] **OBS-056** Create alert: AgentHighLatency
  - **Condition:** P95 latency > 500ms for 5 minutes
  - **Severity:** Warning
  - **Acceptance:** Alert fires

- [ ] **OBS-057** Create alert: AgentPodDown
  - **Condition:** Pod unavailable > 2 minutes
  - **Severity:** Critical
  - **Acceptance:** Alert fires

- [ ] **OBS-058** Create alert: AgentHPAMaxReplicas
  - **Condition:** At max replicas for 15 minutes
  - **Severity:** Warning
  - **Acceptance:** Alert fires

- [ ] **OBS-059** Create alert: DatabaseConnectionErrors
  - **Condition:** Connection errors > 0 for 1 minute
  - **Severity:** Critical
  - **Acceptance:** Alert fires

- [ ] **OBS-060** Create alert: RedisHighMemory
  - **Condition:** Memory usage > 90%
  - **Severity:** Warning
  - **Acceptance:** Alert fires

- [ ] **OBS-061** Create alert: CertificateExpiringSoon
  - **Condition:** Certificate expires < 30 days
  - **Severity:** Warning
  - **Acceptance:** Alert fires

- [ ] **OBS-062** Create EUDR-specific alerts (stricter)
  - **Error Rate:** > 0.5%
  - **Latency:** P95 > 250ms
  - **Acceptance:** Stricter alerts for critical agent

## 4.5 Alerting Integration (8 Tasks)

- [ ] **OBS-063** Configure Alertmanager receivers
  - **Slack:** #alerts-warning, #alerts-critical
  - **PagerDuty:** Production service
  - **Email:** DevOps team
  - **Acceptance:** Receivers configured

- [ ] **OBS-064** Configure routing by severity
  - **Critical:** PagerDuty + Slack
  - **Warning:** Slack only
  - **Acceptance:** Routing works

- [ ] **OBS-065** Configure alert grouping
  - **Group By:** alertname, agent
  - **Group Wait:** 30s
  - **Acceptance:** Alerts grouped

- [ ] **OBS-066** Configure alert repeat interval
  - **Interval:** 4 hours
  - **Acceptance:** No alert spam

- [ ] **OBS-067** Create PagerDuty service
  - **Service:** GreenLang Agents
  - **Acceptance:** Service created

- [ ] **OBS-068** Configure escalation policy
  - **Level 1:** Primary on-call (5 min)
  - **Level 2:** Secondary on-call (15 min)
  - **Acceptance:** Escalation configured

- [ ] **OBS-069** Create incident response runbook
  - **File:** `docs/runbooks/incident-response.md`
  - **Acceptance:** Runbook complete

- [ ] **OBS-070** Test alert delivery end-to-end
  - **Method:** Fire test alert
  - **Acceptance:** Alert received

---

# SECTION 5: DISASTER RECOVERY (40 Tasks)

## 5.1 Backup Automation (15 Tasks)

### 5.1.1 Velero Backup
- [ ] **DR-001** Deploy Velero to Kubernetes
  - **Helm:** vmware-tanzu/velero
  - **Acceptance:** Velero running

- [ ] **DR-002** Configure AWS S3 backup location
  - **Bucket:** greenlang-velero-backups
  - **Acceptance:** Backup location configured

- [ ] **DR-003** Configure snapshot location (EBS)
  - **Provider:** AWS
  - **Acceptance:** EBS snapshots enabled

- [ ] **DR-004** Create backup schedule (daily 2 AM)
  - **Schedule:** `0 2 * * *`
  - **Acceptance:** Daily backups running

- [ ] **DR-005** Configure backup TTL (30 days)
  - **TTL:** 720h
  - **Acceptance:** Old backups deleted

- [ ] **DR-006** Test backup creation
  - **Command:** `velero backup create test-backup`
  - **Acceptance:** Backup completes

- [ ] **DR-007** Verify backup integrity
  - **Command:** `velero backup describe test-backup`
  - **Acceptance:** Backup verified

### 5.1.2 Database Backup
- [ ] **DR-008** Configure RDS automated snapshots
  - **Window:** 03:00-04:00 UTC
  - **Acceptance:** Snapshots automated

- [ ] **DR-009** Set snapshot retention (30 days)
  - **Retention:** 30 days
  - **Acceptance:** Retention configured

- [ ] **DR-010** Configure point-in-time recovery
  - **Retention:** 7 days
  - **Acceptance:** PITR enabled

- [ ] **DR-011** Enable cross-region snapshot copy
  - **Destination:** us-west-2
  - **Acceptance:** Snapshots copied to DR

- [ ] **DR-012** Test database restore
  - **Method:** Restore to test instance
  - **Acceptance:** Data verified

### 5.1.3 Redis Backup
- [ ] **DR-013** Configure ElastiCache snapshots
  - **Window:** 05:00-06:00 UTC
  - **Acceptance:** Snapshots automated

- [ ] **DR-014** Set snapshot retention (7 days)
  - **Retention:** 7 days
  - **Acceptance:** Retention configured

- [ ] **DR-015** Test Redis restore
  - **Method:** Restore to test cluster
  - **Acceptance:** Data verified

## 5.2 Restore Procedures (12 Tasks)

### 5.2.1 Kubernetes Restore
- [ ] **DR-016** Document Velero restore procedure
  - **File:** `docs/runbooks/velero-restore.md`
  - **Acceptance:** Procedure documented

- [ ] **DR-017** Create restore validation checklist
  - **Checks:** Pods running, services accessible, data intact
  - **Acceptance:** Checklist created

- [ ] **DR-018** Test restore to new cluster
  - **Target:** DR cluster
  - **Acceptance:** Restore successful

- [ ] **DR-019** Measure restore time (RTO target: 1h)
  - **Measurement:** Full restore duration
  - **Acceptance:** RTO < 1 hour

- [ ] **DR-020** Document common restore issues
  - **File:** `docs/runbooks/restore-troubleshooting.md`
  - **Acceptance:** Issues documented

### 5.2.2 Database Restore
- [ ] **DR-021** Document RDS snapshot restore
  - **File:** `docs/runbooks/rds-restore.md`
  - **Acceptance:** Procedure documented

- [ ] **DR-022** Document point-in-time recovery
  - **File:** `docs/runbooks/rds-pitr.md`
  - **Acceptance:** PITR documented

- [ ] **DR-023** Create restore validation queries
  - **Queries:** Data integrity checks
  - **Acceptance:** Validation queries ready

- [ ] **DR-024** Test restore to test instance
  - **Method:** Snapshot restore
  - **Acceptance:** Restore verified

### 5.2.3 Redis Restore
- [ ] **DR-025** Document ElastiCache restore procedure
  - **File:** `docs/runbooks/redis-restore.md`
  - **Acceptance:** Procedure documented

- [ ] **DR-026** Create cache validation procedure
  - **Checks:** Key existence, data integrity
  - **Acceptance:** Validation procedure

- [ ] **DR-027** Document cache warming procedure
  - **File:** `docs/runbooks/cache-warming.md`
  - **Acceptance:** Warming documented

## 5.3 Failover Testing (13 Tasks)

### 5.3.1 DR Drill Procedure
- [ ] **DR-028** Create DR drill runbook
  - **File:** `docs/runbooks/dr-drill.md`
  - **Acceptance:** Runbook complete

- [ ] **DR-029** Define drill success criteria
  - **Criteria:** RTO < 1h, RPO < 15min, all services healthy
  - **Acceptance:** Criteria defined

- [ ] **DR-030** Schedule quarterly DR drills
  - **Schedule:** First week of Q1, Q2, Q3, Q4
  - **Acceptance:** Drills scheduled

- [ ] **DR-031** Create communication templates
  - **Templates:** Drill start, drill complete, incident
  - **Acceptance:** Templates created

- [ ] **DR-032** Define rollback procedure
  - **Procedure:** Failback to primary
  - **Acceptance:** Rollback defined

### 5.3.2 Execute DR Drill
- [ ] **DR-033** Notify stakeholders of drill
  - **Stakeholders:** Engineering, Product, Support
  - **Acceptance:** Notification sent

- [ ] **DR-034** Simulate primary region failure
  - **Method:** DNS switch, cluster drain
  - **Acceptance:** Primary offline

- [ ] **DR-035** Execute failover to DR region
  - **Script:** `./scripts/failover-to-dr.sh`
  - **Acceptance:** DR active

- [ ] **DR-036** Verify service health in DR
  - **Checks:** All pods running, endpoints responding
  - **Acceptance:** Services healthy

- [ ] **DR-037** Run smoke tests in DR
  - **Script:** `./scripts/smoke-test.sh dr`
  - **Acceptance:** Smoke tests pass

- [ ] **DR-038** Measure RTO and RPO achieved
  - **RTO Target:** < 1 hour
  - **RPO Target:** < 15 minutes
  - **Acceptance:** Targets met

- [ ] **DR-039** Document drill results
  - **Report:** Drill results with improvement areas
  - **Acceptance:** Report complete

- [ ] **DR-040** Create improvement plan
  - **Plan:** Address gaps identified
  - **Acceptance:** Plan documented

---

# SUCCESS METRICS

## CI/CD Metrics
| Metric | Target | Current |
|--------|--------|---------|
| Build Time | < 5 minutes | TBD |
| PR Validation Time | < 10 minutes | TBD |
| Deployment Frequency | > 5/day | TBD |
| Deployment Success Rate | > 95% | TBD |
| Rollback Time | < 5 minutes | TBD |

## Infrastructure Metrics
| Metric | Target | Current |
|--------|--------|---------|
| Infrastructure Drift | 0% | TBD |
| Terraform Apply Time | < 30 minutes | TBD |
| Environment Parity | 100% | TBD |
| IaC Coverage | 100% | TBD |

## Kubernetes Metrics
| Metric | Target | Current |
|--------|--------|---------|
| Cluster Uptime | 99.9% | TBD |
| Pod Startup Time | < 30 seconds | TBD |
| HPA Response Time | < 60 seconds | TBD |
| Resource Utilization | 70% average | TBD |

## Observability Metrics
| Metric | Target | Current |
|--------|--------|---------|
| Metrics Coverage | 100% | TBD |
| Alert Response Time | < 5 minutes | TBD |
| Dashboard Coverage | All services | TBD |
| Log Retention | 30 days | TBD |

## DR Metrics
| Metric | Target | Current |
|--------|--------|---------|
| RTO | < 1 hour | TBD |
| RPO | < 15 minutes | TBD |
| Backup Success Rate | 100% | TBD |
| DR Drill Frequency | Quarterly | TBD |

---

# TASK SUMMARY

| Section | Task Count | Priority |
|---------|------------|----------|
| CI/CD Pipeline | 85 | P0 |
| Infrastructure as Code | 95 | P0 |
| Kubernetes Operations | 75 | P0 |
| Observability Stack | 70 | P1 |
| Disaster Recovery | 40 | P1 |
| **Total** | **365** | |

---

# DEPENDENCIES

```
Phase 1: CI/CD (Week 1-3)
  |
  v
Phase 2: IaC (Week 4-7) --+
  |                       |
  v                       v
Phase 3: K8s Ops       Phase 4: Observability
(Week 5-8)             (Week 8-10)
  |                       |
  +-------+-------+-------+
          |
          v
    Phase 5: DR (Week 10-11)
```

---

# SIGN-OFF

## Phase 1 - CI/CD Completion
- [ ] DevOps Lead: _________________ Date: _______
- [ ] Engineering Lead: _________________ Date: _______

## Phase 2 - IaC Completion
- [ ] DevOps Lead: _________________ Date: _______
- [ ] Security Engineer: _________________ Date: _______

## Phase 3 - Kubernetes Completion
- [ ] DevOps Lead: _________________ Date: _______
- [ ] SRE Lead: _________________ Date: _______

## Phase 4 - Observability Completion
- [ ] DevOps Lead: _________________ Date: _______
- [ ] SRE Lead: _________________ Date: _______

## Phase 5 - DR Completion
- [ ] DevOps Lead: _________________ Date: _______
- [ ] CTO: _________________ Date: _______

---

**Document Control:**
| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-04 | GL-DevOpsEngineer | Initial comprehensive DevOps TODO |

---

**End of Document**