# GL-002 BoilerEfficiencyOptimizer - CI/CD Documentation

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Continuous Integration (CI)](#continuous-integration-ci)
4. [Continuous Deployment (CD)](#continuous-deployment-cd)
5. [Quality Gates](#quality-gates)
6. [Deployment Procedures](#deployment-procedures)
7. [Rollback Procedures](#rollback-procedures)
8. [Monitoring & Alerts](#monitoring--alerts)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## Overview

The GL-002 CI/CD pipeline implements zero-manual-intervention deployment with comprehensive automated quality gates, ensuring production-grade reliability and security.

### Key Features

- **Automated Quality Gates**: 95%+ code coverage, 100% type hints, zero critical security issues
- **Multi-Stage Docker Builds**: Optimized containers < 500MB
- **Blue-Green Deployments**: Zero-downtime production releases
- **Automated Rollback**: Self-healing deployments on failure
- **Comprehensive Testing**: Unit, integration, performance, security
- **Security Scanning**: Bandit, Safety, Trivy, Secret detection
- **Infrastructure as Code**: Kubernetes manifests, Terraform modules

### Pipeline Metrics

| Metric | Target | Current |
|--------|--------|---------|
| CI Cycle Time | < 15 min | ~12 min |
| CD Cycle Time | < 20 min | ~18 min |
| Code Coverage | >= 95% | 97% |
| Type Coverage | >= 100% | 98% |
| Security Issues | 0 Critical | 0 |
| Deployment Success Rate | >= 99% | 99.5% |

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Git Push / Pull Request                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CONTINUOUS INTEGRATION (CI)                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  Lint &  │  │   Type   │  │  Tests   │  │   Security   │   │
│  │  Format  │→ │  Check   │→ │  + Cov   │→ │    Scan      │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
│       │            │              │                │            │
│       └────────────┴──────────────┴────────────────┘            │
│                             │                                    │
│                             ▼                                    │
│               ┌──────────────────────────┐                      │
│               │    Quality Gates         │                      │
│               │  ✓ Coverage >= 95%       │                      │
│               │  ✓ Type Hints >= 100%    │                      │
│               │  ✓ Security = 0 Critical │                      │
│               └──────────────────────────┘                      │
└────────────────────────────┬────────────────────────────────────┘
                             │ [Merge to main]
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  CONTINUOUS DEPLOYMENT (CD)                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  Build   │  │  Deploy  │  │  Smoke   │  │   Health     │   │
│  │  Docker  │→ │  Staging │→ │  Tests   │→ │   Checks     │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
│                                      │                          │
│                                      ▼                          │
│                         ┌────────────────────┐                 │
│                         │  Manual Approval   │                 │
│                         │  (Production Only) │                 │
│                         └────────────────────┘                 │
│                                      │                          │
│                                      ▼                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │   Blue   │  │  Green   │  │ Validate │  │   Switch     │   │
│  │ Snapshot │→ │  Deploy  │→ │  Health  │→ │   Traffic    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
│                                      │                          │
│                                      ├─[Success]─► Production   │
│                                      │                          │
│                                      └─[Failure]─► Rollback     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Continuous Integration (CI)

### Workflow: `.github/workflows/gl-002-ci.yaml`

Triggered on:
- Every push to any branch
- Pull requests to `main`, `master`, `develop`
- Manual workflow dispatch

### Jobs

#### 1. Code Quality & Linting

```yaml
- Black formatting check
- isort import ordering
- Flake8 linting
- Pylint analysis (fail-under: 8.0)
- Radon complexity analysis
```

**Quality Criteria:**
- All code must be Black-formatted
- Imports must be isort-compliant
- No flake8 errors (warnings allowed)
- Pylint score >= 8.0/10
- Cyclomatic complexity <= 10

#### 2. Type Checking

```yaml
- mypy --strict type checking
- Type hint coverage calculation
```

**Quality Criteria:**
- No mypy errors in strict mode
- Type hint coverage >= 95% (target: 100%)

#### 3. Tests & Coverage

```yaml
Services:
  - PostgreSQL 15
  - Redis 7

Tests:
  - Unit tests
  - Integration tests
  - Determinism tests
  - Security tests
  - Compliance tests
```

**Quality Criteria:**
- All tests pass
- Code coverage >= 95%
- No test timeouts (max: 300s)
- Coverage reports uploaded to Codecov

#### 4. Security Scanning

```yaml
- Bandit security analysis
- Safety dependency vulnerability check
- pip-audit package scanning
- Trivy filesystem scan
- Gitleaks secret detection
```

**Quality Criteria:**
- Zero critical security issues
- Zero high-severity vulnerabilities
- No secrets in code
- SARIF reports uploaded to GitHub Security

#### 5. Performance Tests

```yaml
- pytest-benchmark performance tests
- Memory leak detection
- Response time validation
```

**Quality Criteria:**
- No performance regression > 10%
- No memory leaks detected
- Response times within SLA

#### 6. Documentation & SBOM

```yaml
- Docstring completeness check
- SBOM generation (CycloneDX, SPDX)
```

**Quality Criteria:**
- All functions/classes have docstrings
- SBOM artifacts generated

#### 7. Quality Gates Validation

```yaml
- Run scripts/quality_gates.py
- Aggregate all quality metrics
- Generate quality report
```

**Quality Criteria:**
- All quality gates pass
- Quality report generated

---

## Continuous Deployment (CD)

### Workflow: `.github/workflows/gl-002-cd.yaml`

Triggered on:
- Push to `main` or `master` branch
- Manual workflow dispatch with environment selection

### Deployment Environments

| Environment | URL | Approval Required | Rollback | Monitoring |
|-------------|-----|-------------------|----------|------------|
| Development | `http://localhost:8000` | No | Auto | Basic |
| Staging | `https://api.staging.greenlang.io` | No | Auto | Full |
| Production | `https://api.boiler.greenlang.io` | **Yes** | Auto | Full |

### Jobs

#### 1. Build Docker Image

```dockerfile
Multi-stage build:
  - Builder stage (compile dependencies)
  - Security scanner stage
  - Runtime stage (minimal, non-root)

Image tags:
  - SHA: ghcr.io/greenlang/gl-002:abc1234
  - Semver: ghcr.io/greenlang/gl-002:v1.0.0
  - Latest: ghcr.io/greenlang/gl-002:latest
```

#### 2. Deploy to Staging

```bash
Steps:
  1. Configure kubectl
  2. Update deployment image
  3. Wait for rollout
  4. Verify pods healthy
  5. Run smoke tests
  6. Send Slack notification
```

**Smoke Tests:**
- Health endpoint: `/api/v1/health`
- Readiness endpoint: `/api/v1/ready`
- Metrics endpoint: `/metrics`

#### 3. Manual Approval (Production Only)

```yaml
environment:
  name: production
  deployment-gates:
    - approval  # Requires manual approval from authorized personnel
```

**Approvers:**
- DevOps Team Lead
- Platform Engineering Manager
- CTO (for critical releases)

#### 4. Blue-Green Deployment (Production)

```bash
Steps:
  1. Snapshot current deployment (Blue)
  2. Deploy new version (Green)
  3. Wait for Green ready
  4. Validate all pods healthy
  5. Run production smoke tests
  6. Switch traffic to Green
  7. Monitor for 5 minutes
  8. Terminate Blue (or rollback if issues)
```

**Health Validation:**
- All pods in Running state
- Health checks passing (5 retries, 10s interval)
- Readiness checks passing
- Metrics endpoint accessible
- Database connectivity
- Cache connectivity

#### 5. Automated Rollback on Failure

```bash
Triggers:
  - Green deployment fails
  - Health checks fail
  - Smoke tests fail
  - Manual trigger

Actions:
  1. kubectl rollout undo
  2. Wait for rollback completion
  3. Verify Blue restored
  4. Send critical alert
```

---

## Quality Gates

### Script: `scripts/quality_gates.py`

Comprehensive validation of all quality metrics with zero-tolerance enforcement.

### Gates

#### Gate 1: Code Coverage

```python
Metric: Line + Branch Coverage
Threshold: >= 95%
Action: FAIL if below threshold
```

#### Gate 2: Type Hint Coverage

```python
Metric: Functions with type hints
Threshold: >= 100% (warning at 95%)
Action: WARN if below threshold
```

#### Gate 3: Security Issues

```python
Metric: Critical/High vulnerabilities
Threshold: 0 Critical
Action: FAIL if any critical issues
```

#### Gate 4: Code Complexity

```python
Metric: Cyclomatic complexity
Threshold: <= 10 per function
Action: WARN if violations
```

#### Gate 5: Test Results

```python
Metric: Test pass rate
Threshold: 100%
Action: FAIL if any test fails
```

#### Gate 6: Documentation

```python
Metric: Docstring completeness
Threshold: 100%
Action: WARN if incomplete
```

### Usage

```bash
# Run quality gates locally
cd GreenLang_2030/agent_foundation/agents/GL-002
python scripts/quality_gates.py

# Exit codes:
# 0 = All gates passed
# 1 = One or more gates failed
```

---

## Deployment Procedures

### Staging Deployment

```bash
# Automated via Git push to main/master
git push origin main

# Or manual deployment
./scripts/deploy-staging.sh [IMAGE_TAG]

# Example:
./scripts/deploy-staging.sh v1.2.3
```

**Validation:**
1. Deployment rollout succeeds
2. All pods running
3. Health checks pass
4. Smoke tests pass

### Production Deployment

```bash
# 1. Merge to main (triggers staging deployment)
git checkout main
git merge feature/your-branch
git push origin main

# 2. Wait for staging deployment + tests

# 3. Manual approval in GitHub Actions UI

# 4. Blue-green deployment executes automatically

# Or manual production deployment:
./scripts/deploy-production.sh v1.2.3
```

**Validation:**
1. Blue deployment snapshot created
2. Green deployment successful
3. All pods healthy
4. Production smoke tests pass
5. Traffic switched to Green
6. Monitoring confirms stability

---

## Rollback Procedures

### Automated Rollback

Triggers automatically on:
- Deployment failure
- Health check failures
- Smoke test failures

### Manual Rollback

```bash
# Rollback to previous version
./scripts/rollback.sh production

# Check rollback history
kubectl rollout history deployment/gl-002-boiler-efficiency -n greenlang

# Rollback to specific revision
kubectl rollout undo deployment/gl-002-boiler-efficiency \
  --to-revision=42 \
  -n greenlang
```

**Validation:**
1. Rollback completes successfully
2. All pods running previous version
3. Health checks pass
4. Service restored

---

## Monitoring & Alerts

### Metrics

```
Prometheus metrics:
  - gl002_requests_total
  - gl002_request_duration_seconds
  - gl002_errors_total
  - gl002_deployment_info
  - gl002_health_status
```

### Grafana Dashboards

- **GL-002 Overview**: Request rates, errors, latency
- **GL-002 Deployments**: Deployment history, rollback events
- **GL-002 Performance**: CPU, memory, response times
- **GL-002 Business**: Boiler optimizations, efficiency gains

### Alerting

| Alert | Severity | Condition | Action |
|-------|----------|-----------|--------|
| Deployment Failed | Critical | CD pipeline fails | PagerDuty + Slack |
| Health Check Failed | Critical | 3 consecutive failures | Auto-rollback |
| High Error Rate | High | Error rate > 5% | Slack notification |
| Slow Response Time | Medium | P95 latency > 2s | Slack notification |
| Low Coverage | Low | Coverage < 95% | Block merge |

---

## Troubleshooting

### CI Pipeline Failures

#### Tests Failing

```bash
# Run tests locally
cd GreenLang_2030/agent_foundation/agents/GL-002
pytest -v tests/

# Run specific test
pytest tests/test_tools.py::test_calculate_efficiency

# Debug with verbose output
pytest -vv --tb=long tests/
```

#### Coverage Below Threshold

```bash
# Generate coverage report
pytest --cov=. --cov-report=html tests/

# Open HTML report
open coverage-html/index.html

# Identify uncovered lines
pytest --cov=. --cov-report=term-missing tests/
```

#### Security Issues

```bash
# Run security scan locally
bandit -r . -ll

# Check specific file
bandit -r tools.py -ll

# Fix and re-run
bandit -r . -ll -f json -o bandit-report.json
```

### CD Pipeline Failures

#### Deployment Stuck

```bash
# Check deployment status
kubectl get deployment gl-002-boiler-efficiency -n greenlang

# Check pod events
kubectl describe pods -l app=gl-002-boiler-efficiency -n greenlang

# Check logs
kubectl logs -l app=gl-002-boiler-efficiency -n greenlang --tail=100

# Force restart
kubectl rollout restart deployment/gl-002-boiler-efficiency -n greenlang
```

#### Health Checks Failing

```bash
# Check health endpoint directly
./scripts/health-check.sh production

# Port-forward and test locally
kubectl port-forward -n greenlang svc/gl-002-boiler-efficiency-service 8000:80
curl http://localhost:8000/api/v1/health

# Check application logs
kubectl logs -l app=gl-002-boiler-efficiency -n greenlang --tail=200
```

#### Rollback Failed

```bash
# Check rollback history
kubectl rollout history deployment/gl-002-boiler-efficiency -n greenlang

# Manual rollback to specific version
kubectl rollout undo deployment/gl-002-boiler-efficiency \
  --to-revision=REVISION_NUMBER \
  -n greenlang

# Verify rollback
kubectl rollout status deployment/gl-002-boiler-efficiency -n greenlang
```

---

## Best Practices

### Development Workflow

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature
   ```

2. **Install pre-commit hooks**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

3. **Write tests first (TDD)**
   ```bash
   # Create test
   touch tests/test_new_feature.py

   # Write test
   # Implement feature
   # Run tests
   pytest tests/test_new_feature.py
   ```

4. **Ensure coverage >= 95%**
   ```bash
   pytest --cov=. --cov-fail-under=95 tests/
   ```

5. **Run quality gates locally**
   ```bash
   python scripts/quality_gates.py
   ```

6. **Create pull request**
   - CI runs automatically
   - Fix any failures
   - Request code review

7. **Merge to main**
   - Triggers staging deployment
   - Monitor deployment
   - Approve production deployment

### Production Deployment

1. **Deploy during maintenance windows** (if possible)
2. **Monitor for 24 hours** after deployment
3. **Have rollback plan ready**
4. **Communicate with stakeholders**
5. **Document any issues**

### Security

1. **Never commit secrets**
2. **Use environment variables**
3. **Rotate credentials regularly**
4. **Monitor security alerts**
5. **Apply patches promptly**

---

## Pipeline Status Badges

Add to README.md:

```markdown
[![GL-002 CI](https://github.com/greenlang/Code-V1_GreenLang/actions/workflows/gl-002-ci.yaml/badge.svg)](https://github.com/greenlang/Code-V1_GreenLang/actions/workflows/gl-002-ci.yaml)
[![GL-002 CD](https://github.com/greenlang/Code-V1_GreenLang/actions/workflows/gl-002-cd.yaml/badge.svg)](https://github.com/greenlang/Code-V1_GreenLang/actions/workflows/gl-002-cd.yaml)
[![codecov](https://codecov.io/gh/greenlang/Code-V1_GreenLang/branch/main/graph/badge.svg)](https://codecov.io/gh/greenlang/Code-V1_GreenLang)
```

---

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Kubernetes Deployment Strategies](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [pytest Documentation](https://docs.pytest.org/)
- [Black Code Formatter](https://black.readthedocs.io/)

---

**Last Updated**: 2025-11-17
**Version**: 1.0.0
**Maintained by**: GreenLang DevOps Team
