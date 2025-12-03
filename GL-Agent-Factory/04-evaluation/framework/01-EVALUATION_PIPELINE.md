# Evaluation Pipeline & CI/CD Integration

**Version:** 1.0.0
**Date:** 2025-12-03
**Status:** Active
**Owner:** GreenLang DevOps & Quality Engineering Team

---

## Executive Summary

This document defines the automated evaluation pipeline that integrates with CI/CD to ensure every agent change is automatically tested, validated, and certified before deployment. The pipeline enforces quality gates at every stage, from commit to production, with automated evaluation on every change and manual review checkpoints for certification.

**Core Principle:** Automate quality enforcement, not quality assurance.

---

## Pipeline Overview

### Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────────┐
│                         EVALUATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [Commit] → [Unit Tests] → [Golden Tests] → [Performance Tests]    │
│                    ↓              ↓                ↓                 │
│                   PASS          PASS             PASS                │
│                    ↓              ↓                ↓                 │
│            [Pull Request] → [Integration Tests] → [Security Scan]   │
│                    ↓              ↓                ↓                 │
│                   PASS          PASS             PASS                │
│                    ↓              ↓                ↓                 │
│              [Pre-Release] → [Compliance Tests] → [Climate Science] │
│                    ↓              ↓                ↓                 │
│                   PASS          PASS             PASS                │
│                    ↓              ↓                ↓                 │
│            [Certification] → [Manual Review] → [Final Approval]     │
│                    ↓              ↓                ↓                 │
│               CERTIFIED       CERTIFIED        CERTIFIED             │
│                    ↓              ↓                ↓                 │
│              [Deployment] → [Staging] → [Production]                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Commit (Automated)

### Trigger

Every commit to any branch.

### Automated Checks

1. **Linting & Formatting**
   - Tool: flake8, black, isort
   - Pass Criteria: No linting errors
   - Duration: ~30 seconds

2. **Type Checking**
   - Tool: mypy
   - Pass Criteria: No type errors
   - Duration: ~1 minute

3. **Unit Tests (Core)**
   - Tool: pytest
   - Tests: Core unit tests only (fast subset)
   - Pass Criteria: 100% pass rate
   - Duration: ~2 minutes

4. **Security Scan (Quick)**
   - Tool: bandit
   - Pass Criteria: No P0 vulnerabilities
   - Duration: ~1 minute

### GitHub Actions Workflow

```yaml
# .github/workflows/commit-checks.yml

name: Commit Checks

on:
  push:
    branches: ['**']
  pull_request:
    branches: [main, develop]

jobs:
  commit-checks:
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Lint with flake8
      run: |
        flake8 greenlang/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 greenlang/ tests/ --count --max-complexity=10 --max-line-length=100 --statistics

    - name: Format check with black
      run: |
        black --check greenlang/ tests/

    - name: Import sorting check with isort
      run: |
        isort --check-only greenlang/ tests/

    - name: Type check with mypy
      run: |
        mypy greenlang/ --ignore-missing-imports --no-strict-optional

    - name: Run unit tests (core)
      run: |
        pytest tests/ \
          -m "not slow and not integration and not performance" \
          --maxfail=5 \
          --tb=short \
          --duration=10

    - name: Security scan with bandit
      run: |
        bandit -r greenlang/ -ll -i -x tests/

    - name: Report status
      if: always()
      run: |
        echo "Commit checks completed"
        echo "Status: ${{ job.status }}"
```

### Notification

- **Pass:** Green checkmark on commit, no notification
- **Fail:** Red X on commit, Slack notification to developer

---

## Stage 2: Pull Request (Automated + Manual Review)

### Trigger

Pull request opened or updated.

### Automated Checks

1. **Full Test Suite**
   - Tool: pytest
   - Tests: All unit tests + integration tests
   - Pass Criteria: 100% pass rate, >85% coverage
   - Duration: ~10 minutes

2. **Golden Tests**
   - Tool: pytest
   - Tests: All 25+ golden test scenarios
   - Pass Criteria: 100% pass rate
   - Duration: ~5 minutes

3. **Performance Regression Tests**
   - Tool: pytest-benchmark
   - Tests: Latency and cost benchmarks
   - Pass Criteria: No regression >10% vs. baseline
   - Duration: ~5 minutes

4. **Code Quality Scan**
   - Tool: SonarQube
   - Pass Criteria: Quality gate score >8.0/10
   - Duration: ~3 minutes

5. **Security Scan (Full)**
   - Tool: bandit, safety, pip-audit
   - Pass Criteria: No P0/P1 vulnerabilities, no critical CVEs
   - Duration: ~2 minutes

### Manual Review

- **Code Review:** 2 approvals required (1 senior engineer + 1 domain expert)
- **Review Checklist:**
  - [ ] Code implements specification correctly
  - [ ] Error handling comprehensive
  - [ ] Provenance tracking complete
  - [ ] Test coverage >85%
  - [ ] Golden tests passing
  - [ ] Documentation updated

### GitHub Actions Workflow

```yaml
# .github/workflows/pull-request-checks.yml

name: Pull Request Checks

on:
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 30

    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run full test suite with coverage
      run: |
        pytest tests/ \
          --cov=greenlang \
          --cov-report=xml \
          --cov-report=html \
          --cov-report=term-missing \
          --cov-fail-under=85 \
          --tb=short \
          --duration=20

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml
        flags: unittests-${{ matrix.python-version }}
        name: codecov-${{ matrix.python-version }}

  golden-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 15

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run golden tests
      run: |
        pytest tests/ \
          -m "golden" \
          --verbose \
          --tb=short

    - name: Validate golden test pass rate
      run: |
        # Extract pass rate from pytest output
        # Assert 100% pass rate
        python scripts/validate_golden_tests.py

  performance:
    runs-on: ubuntu-latest
    timeout-minutes: 20

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run performance benchmarks
      run: |
        pytest tests/ \
          -m "performance" \
          --benchmark-only \
          --benchmark-autosave \
          --benchmark-compare=baseline

    - name: Check for performance regression
      run: |
        python scripts/check_performance_regression.py \
          --threshold=0.10 \
          --baseline=benchmarks/.benchmarks/baseline.json \
          --current=benchmarks/.benchmarks/latest.json

  security:
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        pip install bandit safety pip-audit

    - name: Security scan with bandit
      run: |
        bandit -r greenlang/ \
          -ll \
          -f json \
          -o bandit-report.json

    - name: Dependency vulnerability scan with safety
      run: |
        safety check \
          --json \
          --output safety-report.json

    - name: Dependency audit with pip-audit
      run: |
        pip-audit \
          --format json \
          --output pip-audit-report.json

    - name: Validate security scan results
      run: |
        python scripts/validate_security_scan.py \
          --bandit bandit-report.json \
          --safety safety-report.json \
          --pip-audit pip-audit-report.json

    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json
          pip-audit-report.json

  code-quality:
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
    - uses: actions/checkout@v3

    - name: SonarQube Scan
      uses: sonarsource/sonarqube-scan-action@master
      env:
        SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
        SONAR_HOST_URL: ${{ secrets.SONAR_HOST_URL }}

    - name: SonarQube Quality Gate
      uses: sonarsource/sonarqube-quality-gate-action@master
      timeout-minutes: 5
      env:
        SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}

  pr-checks-summary:
    runs-on: ubuntu-latest
    needs: [test, golden-tests, performance, security, code-quality]
    if: always()

    steps:
    - name: Check all jobs passed
      run: |
        if [ "${{ needs.test.result }}" != "success" ] || \
           [ "${{ needs.golden-tests.result }}" != "success" ] || \
           [ "${{ needs.performance.result }}" != "success" ] || \
           [ "${{ needs.security.result }}" != "success" ] || \
           [ "${{ needs.code-quality.result }}" != "success" ]; then
          echo "One or more checks failed"
          exit 1
        fi
        echo "All PR checks passed!"
```

### Notification

- **Pass:** Green checkmark on PR, ready for code review
- **Fail:** Red X on PR, Slack notification with failure details

---

## Stage 3: Pre-Release (Automated + Manual Review)

### Trigger

Release branch created (e.g., `release/v1.0.0`) or release tag pushed.

### Automated Checks

1. **Comprehensive Golden Tests**
   - Tool: pytest
   - Tests: 25+ golden tests + cross-platform validation
   - Pass Criteria: 100% pass rate on Windows, Linux, macOS
   - Duration: ~15 minutes

2. **Load Testing**
   - Tool: locust
   - Tests: Steady state (100 req/s, 1 hour), spike (500 req/s, 5 min)
   - Pass Criteria: Throughput meets targets, error rate <1%
   - Duration: ~1 hour 10 minutes

3. **Integration Testing**
   - Tool: pytest
   - Tests: Agent-to-agent integration, external API integration
   - Pass Criteria: 100% pass rate
   - Duration: ~10 minutes

4. **Compliance Tests**
   - Tool: pytest
   - Tests: Regulatory methodology validation (CBAM, CSRD, EPA)
   - Pass Criteria: 100% pass rate
   - Duration: ~5 minutes

5. **Documentation Validation**
   - Tool: sphinx, markdownlint
   - Pass Criteria: All docs build successfully, no broken links
   - Duration: ~3 minutes

### Manual Review

- **Legal Review:** Compliance with regulations (if applicable)
- **Climate Science Review:** Emission factors, thermodynamics validation (if applicable)
- **Release Notes:** Reviewed and approved by Product Manager

### GitHub Actions Workflow

```yaml
# .github/workflows/pre-release-checks.yml

name: Pre-Release Checks

on:
  push:
    branches:
      - 'release/**'
    tags:
      - 'v*.*.*'

jobs:
  golden-tests-cross-platform:
    runs-on: ${{ matrix.os }}
    timeout-minutes: 20

    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.11']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run golden tests
      run: |
        pytest tests/ \
          -m "golden" \
          --verbose \
          --tb=short

    - name: Validate determinism across platforms
      run: |
        python scripts/validate_cross_platform_determinism.py

  load-testing:
    runs-on: ubuntu-latest
    timeout-minutes: 90

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        pip install locust

    - name: Start agent server
      run: |
        python scripts/start_agent_server.py &
        sleep 10  # Wait for server to start

    - name: Run steady state load test (100 req/s, 1 hour)
      run: |
        locust -f benchmarks/locustfile_boiler_efficiency.py \
          --host=http://localhost:8080 \
          --users=100 \
          --spawn-rate=10 \
          --run-time=1h \
          --headless \
          --html=reports/locust-steady-state.html \
          --csv=reports/locust-steady-state

    - name: Run spike load test (500 req/s, 5 min)
      run: |
        locust -f benchmarks/locustfile_boiler_efficiency.py \
          --host=http://localhost:8080 \
          --users=500 \
          --spawn-rate=100 \
          --run-time=5m \
          --headless \
          --html=reports/locust-spike.html \
          --csv=reports/locust-spike

    - name: Validate load test results
      run: |
        python scripts/validate_load_test_results.py \
          --steady-state reports/locust-steady-state_stats.csv \
          --spike reports/locust-spike_stats.csv

    - name: Upload load test reports
      uses: actions/upload-artifact@v3
      with:
        name: load-test-reports
        path: reports/

  integration-testing:
    runs-on: ubuntu-latest
    timeout-minutes: 15

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run integration tests
      run: |
        pytest tests/ \
          -m "integration" \
          --verbose \
          --tb=short

  compliance-testing:
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run compliance tests
      run: |
        pytest tests/ \
          -m "compliance" \
          --verbose \
          --tb=short

  documentation:
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install sphinx sphinx-rtd-theme

    - name: Build documentation
      run: |
        cd docs
        make html

    - name: Check for broken links
      run: |
        cd docs
        make linkcheck

  pre-release-summary:
    runs-on: ubuntu-latest
    needs: [
      golden-tests-cross-platform,
      load-testing,
      integration-testing,
      compliance-testing,
      documentation
    ]
    if: always()

    steps:
    - name: Check all jobs passed
      run: |
        if [ "${{ needs.golden-tests-cross-platform.result }}" != "success" ] || \
           [ "${{ needs.load-testing.result }}" != "success" ] || \
           [ "${{ needs.integration-testing.result }}" != "success" ] || \
           [ "${{ needs.compliance-testing.result }}" != "success" ] || \
           [ "${{ needs.documentation.result }}" != "success" ]; then
          echo "One or more pre-release checks failed"
          exit 1
        fi
        echo "All pre-release checks passed! Ready for certification."
```

### Notification

- **Pass:** Slack notification to QA team: "Agent ready for certification review"
- **Fail:** Slack notification to developer: "Pre-release checks failed"

---

## Stage 4: Certification (Manual Review)

### Trigger

Manual trigger by QA Engineer after pre-release checks pass.

### Manual Review Process

1. **Compliance Review (Legal + Regulatory)**
   - Duration: 3-5 business days
   - Reviewers: Legal Counsel + Regulatory Expert
   - Output: Compliance Review Report (signed)

2. **Climate Science Review (Science Board)**
   - Duration: 3-5 business days
   - Reviewers: 2+ Climate Scientists
   - Output: Climate Science Review Report (signed)

3. **Final Certification Decision (Certification Committee)**
   - Duration: 1-2 business days
   - Committee: VP Engineering, Legal Counsel, Chief Climate Scientist, Product Manager
   - Output: Certification Decision (CERTIFIED / CONDITIONAL / REJECTED)

### Certification Application

```yaml
# certification_application.yml

agent_id: GL-002
agent_name: BoilerEfficiencyOptimizer
version: 1.0.0
applicant: John Doe (john.doe@greenlang.com)
application_date: 2025-12-03

# Pre-Release Check Results
pre_release_checks:
  golden_tests: PASS
  load_testing: PASS
  integration_testing: PASS
  compliance_testing: PASS
  documentation: PASS

# Attachments
attachments:
  - specification: specs/domain1_industrial/boiler/agent_002_boiler_efficiency.yaml
  - implementation: greenlang/agents/boiler_efficiency_optimizer.py
  - test_suite: tests/agents/test_boiler_efficiency_optimizer.py
  - golden_tests: tests/agents/test_boiler_efficiency_optimizer_golden.py
  - benchmarks: benchmarks/boiler_efficiency_optimizer_performance.py
  - validation_report: docs/certification/agent_002_validation_report.md

# Sign-Offs Required
sign_offs:
  - technical_review: PENDING (QA Engineer)
  - compliance_review: PENDING (Legal Counsel)
  - climate_science_review: PENDING (Climate Scientist)
  - final_decision: PENDING (Certification Committee)
```

### Certification Workflow

```python
# scripts/certification_workflow.py

import yaml
import smtplib
from datetime import datetime
from typing import Dict, Any


class CertificationWorkflow:
    """Automate certification workflow notifications and tracking."""

    def __init__(self, application_file: str):
        with open(application_file, 'r') as f:
            self.application = yaml.safe_load(f)

    def submit_application(self):
        """Submit certification application."""
        # Send email to QA Engineer
        self.send_notification(
            to="qa@greenlang.com",
            subject=f"Certification Application: {self.application['agent_name']}",
            body=f"""
            A new certification application has been submitted:

            Agent: {self.application['agent_name']} (v{self.application['version']})
            Applicant: {self.application['applicant']}
            Date: {self.application['application_date']}

            Pre-Release Checks: All PASS

            Please begin technical review.

            Application Details:
            https://github.com/greenlang/greenlang/blob/main/certification_applications/{self.application['agent_id']}_v{self.application['version']}.yml
            """
        )

    def notify_compliance_review(self):
        """Notify Legal for compliance review."""
        self.send_notification(
            to="legal@greenlang.com",
            subject=f"Compliance Review Required: {self.application['agent_name']}",
            body=f"""
            Technical review complete. Please begin compliance review.

            Agent: {self.application['agent_name']} (v{self.application['version']})

            Documents:
            - Specification: {self.application['attachments'][0]['specification']}
            - Validation Report: {self.application['attachments'][5]['validation_report']}
            """
        )

    def notify_climate_science_review(self):
        """Notify Science Board for climate science review."""
        self.send_notification(
            to="science-board@greenlang.com",
            subject=f"Climate Science Review Required: {self.application['agent_name']}",
            body=f"""
            Technical review complete. Please begin climate science review.

            Agent: {self.application['agent_name']} (v{self.application['version']})

            Documents:
            - Specification: {self.application['attachments'][0]['specification']}
            - Implementation: {self.application['attachments'][1]['implementation']}
            - Golden Tests: {self.application['attachments'][3]['golden_tests']}
            """
        )

    def notify_final_decision(self):
        """Notify Certification Committee for final decision."""
        self.send_notification(
            to="cert-committee@greenlang.com",
            subject=f"Final Certification Decision Required: {self.application['agent_name']}",
            body=f"""
            All reviews complete. Please make final certification decision.

            Agent: {self.application['agent_name']} (v{self.application['version']})

            Review Status:
            - Technical Review: APPROVED
            - Compliance Review: APPROVED
            - Climate Science Review: APPROVED

            Certification Committee Meeting:
            https://calendar.google.com/...
            """
        )

    def send_notification(self, to: str, subject: str, body: str):
        """Send email notification."""
        # Implementation omitted for brevity
        print(f"Email sent to {to}: {subject}")


if __name__ == "__main__":
    workflow = CertificationWorkflow("certification_applications/GL-002_v1.0.0.yml")
    workflow.submit_application()
```

### Notification

- **Application Submitted:** Email to QA Engineer
- **Technical Review Complete:** Email to Legal + Science Board
- **All Reviews Complete:** Email to Certification Committee
- **Decision Made:** Email to applicant + Slack announcement

---

## Stage 5: Deployment (Automated)

### Trigger

Certification Committee approves agent (status = CERTIFIED).

### Automated Deployment

1. **Staging Deployment**
   - Tool: Kubernetes (kubectl)
   - Environment: greenlang-staging
   - Duration: ~5 minutes

2. **Staging Validation**
   - Tool: pytest (smoke tests)
   - Tests: Basic functionality, health checks
   - Duration: ~5 minutes

3. **Production Deployment (Canary)**
   - Tool: Kubernetes (kubectl + Argo Rollouts)
   - Strategy: Canary (10% → 50% → 100% over 1 hour)
   - Duration: ~1 hour

4. **Production Validation**
   - Tool: Prometheus + Grafana
   - Metrics: Latency, error rate, throughput
   - Duration: ~1 hour monitoring

### GitHub Actions Workflow

```yaml
# .github/workflows/deploy-certified-agent.yml

name: Deploy Certified Agent

on:
  workflow_dispatch:
    inputs:
      agent_id:
        description: 'Agent ID (e.g., GL-002)'
        required: true
      version:
        description: 'Version (e.g., 1.0.0)'
        required: true
      environment:
        description: 'Deployment environment'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production

jobs:
  deploy-staging:
    runs-on: ubuntu-latest
    if: github.event.inputs.environment == 'staging'
    timeout-minutes: 15

    steps:
    - uses: actions/checkout@v3

    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'v1.28.0'

    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBE_CONFIG_STAGING }}" | base64 -d > ~/.kube/config

    - name: Deploy to staging
      run: |
        kubectl apply -f kubernetes/${{ github.event.inputs.agent_id }}/deployment.yaml
        kubectl apply -f kubernetes/${{ github.event.inputs.agent_id }}/service.yaml
        kubectl rollout status deployment/${{ github.event.inputs.agent_id }} -n greenlang-staging

    - name: Run smoke tests
      run: |
        pytest tests/smoke/ \
          --agent-id=${{ github.event.inputs.agent_id }} \
          --environment=staging \
          --verbose

  deploy-production:
    runs-on: ubuntu-latest
    if: github.event.inputs.environment == 'production'
    timeout-minutes: 90

    steps:
    - uses: actions/checkout@v3

    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'v1.28.0'

    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBE_CONFIG_PRODUCTION }}" | base64 -d > ~/.kube/config

    - name: Deploy canary (10%)
      run: |
        kubectl argo rollouts set image ${{ github.event.inputs.agent_id }} \
          ${{ github.event.inputs.agent_id }}=greenlang/${{ github.event.inputs.agent_id }}:${{ github.event.inputs.version }} \
          -n greenlang-production

        kubectl argo rollouts set weight ${{ github.event.inputs.agent_id }} 10 -n greenlang-production

    - name: Monitor canary (10 minutes)
      run: |
        sleep 600
        python scripts/validate_canary_metrics.py \
          --agent-id=${{ github.event.inputs.agent_id }} \
          --weight=10

    - name: Promote canary to 50%
      run: |
        kubectl argo rollouts set weight ${{ github.event.inputs.agent_id }} 50 -n greenlang-production

    - name: Monitor canary (20 minutes)
      run: |
        sleep 1200
        python scripts/validate_canary_metrics.py \
          --agent-id=${{ github.event.inputs.agent_id }} \
          --weight=50

    - name: Promote canary to 100%
      run: |
        kubectl argo rollouts promote ${{ github.event.inputs.agent_id }} -n greenlang-production

    - name: Monitor production (30 minutes)
      run: |
        sleep 1800
        python scripts/validate_production_metrics.py \
          --agent-id=${{ github.event.inputs.agent_id }}

    - name: Deployment complete
      run: |
        echo "Production deployment complete: ${{ github.event.inputs.agent_id }} v${{ github.event.inputs.version }}"
```

### Rollback Procedure

```bash
# scripts/rollback_agent.sh

#!/bin/bash

AGENT_ID=$1
PREVIOUS_VERSION=$2

echo "Rolling back agent: $AGENT_ID to version $PREVIOUS_VERSION"

# Rollback in production
kubectl argo rollouts undo $AGENT_ID -n greenlang-production

# Validate rollback
python scripts/validate_production_metrics.py --agent-id=$AGENT_ID

echo "Rollback complete"
```

### Notification

- **Staging Deployed:** Slack notification to QA team
- **Staging Validated:** Slack notification to DevOps team
- **Production Canary (10%):** Slack notification to SRE team
- **Production Canary (50%):** Slack notification to SRE team
- **Production (100%):** Slack announcement to #engineering + #product

---

## Validation Report Generation

### Automated Report Generation

After each stage, generate validation report:

```python
# scripts/generate_validation_report.py

import json
from datetime import datetime
from typing import Dict, Any


class ValidationReportGenerator:
    """Generate validation reports for agents."""

    def __init__(self, agent_id: str, version: str):
        self.agent_id = agent_id
        self.version = version
        self.timestamp = datetime.utcnow().isoformat()

    def generate_report(
        self,
        test_results: Dict[str, Any],
        benchmark_results: Dict[str, Any]
    ) -> str:
        """Generate comprehensive validation report."""
        report = f"""
# Validation Report: {self.agent_id} v{self.version}

**Date:** {self.timestamp}
**Status:** {'PASS' if self.all_checks_passed(test_results) else 'FAIL'}

---

## Test Results

### Unit Tests
- Total: {test_results['unit_tests']['total']}
- Passed: {test_results['unit_tests']['passed']}
- Failed: {test_results['unit_tests']['failed']}
- Coverage: {test_results['unit_tests']['coverage']:.1f}%

### Golden Tests
- Total: {test_results['golden_tests']['total']}
- Passed: {test_results['golden_tests']['passed']}
- Failed: {test_results['golden_tests']['failed']}
- Pass Rate: {test_results['golden_tests']['pass_rate']*100:.1f}%

### Performance Tests
- P95 Latency: {benchmark_results['performance']['p95_latency_seconds']:.3f}s (Target: <4.0s)
- Throughput: {benchmark_results['performance']['throughput_rps']:.1f} req/s (Target: >100)
- Success Rate: {benchmark_results['performance']['success_rate']*100:.1f}%

### Accuracy Tests
- Mean Absolute Error: {benchmark_results['accuracy']['mae']:.2f}%
- RMSE: {benchmark_results['accuracy']['rmse']:.2f}%
- Max Error: {benchmark_results['accuracy']['max_error']:.2f}%

### Cost Tests
- Cost per Analysis: ${benchmark_results['cost']['cost_per_analysis_usd']:.4f} (Target: <$0.15)
- Avg Tokens: {benchmark_results['cost']['avg_total_tokens']:.0f}

---

## Certification Status

### 12-Dimension Assessment

| Dimension | Status | Score |
|-----------|--------|-------|
| 1. Specification Completeness | {'PASS' if test_results['specification'] else 'FAIL'} | {test_results['specification_score']}/10 |
| 2. Code Implementation | {'PASS' if test_results['implementation'] else 'FAIL'} | {test_results['implementation_score']}/10 |
| 3. Test Coverage | {'PASS' if test_results['unit_tests']['coverage'] >= 85 else 'FAIL'} | {test_results['unit_tests']['coverage']:.1f}% |
| 4. Deterministic AI Guarantees | {'PASS' if test_results['determinism'] else 'FAIL'} | {'100%' if test_results['determinism'] else '0%'} |
| 5. Documentation Completeness | {'PASS' if test_results['documentation'] else 'FAIL'} | {test_results['documentation_score']}/10 |
| 6. Compliance & Security | {'PASS' if test_results['compliance'] and test_results['security'] else 'FAIL'} | {test_results['compliance_score']}/10 |
| 7. Deployment Readiness | {'PASS' if test_results['deployment'] else 'FAIL'} | {test_results['deployment_score']}/10 |
| 8. Exit Bar Criteria | {'PASS' if self.check_exit_bar(benchmark_results) else 'FAIL'} | {'PASS' if self.check_exit_bar(benchmark_results) else 'FAIL'} |
| 9. Integration & Coordination | {'PASS' if test_results['integration'] else 'FAIL'} | {test_results['integration_score']}/10 |
| 10. Business Impact & Metrics | {'PASS' if test_results['business_impact'] else 'FAIL'} | {test_results['business_impact_score']}/10 |
| 11. Operational Excellence | {'PASS' if test_results['operations'] else 'FAIL'} | {test_results['operations_score']}/10 |
| 12. Continuous Improvement | {'PASS' if test_results['continuous_improvement'] else 'FAIL'} | {test_results['continuous_improvement_score']}/10 |

**Overall Score:** {self.calculate_overall_score(test_results)}/12 PASS

---

## Recommendation

{'CERTIFIED - Agent approved for production deployment' if self.all_checks_passed(test_results) else 'NOT CERTIFIED - Agent must address issues before deployment'}

---

**Report Generated By:** GL-TestEngineer (Automated)
**Report Date:** {self.timestamp}
        """

        return report

    def all_checks_passed(self, test_results: Dict[str, Any]) -> bool:
        """Check if all tests passed."""
        return all([
            test_results['unit_tests']['coverage'] >= 85,
            test_results['golden_tests']['pass_rate'] == 1.0,
            test_results['determinism'] is True,
            test_results['compliance'] is True,
            test_results['security'] is True
        ])

    def check_exit_bar(self, benchmark_results: Dict[str, Any]) -> bool:
        """Check if exit bar criteria met."""
        return all([
            benchmark_results['performance']['p95_latency_seconds'] < 4.0,
            benchmark_results['performance']['throughput_rps'] >= 100,
            benchmark_results['cost']['cost_per_analysis_usd'] < 0.15,
            benchmark_results['accuracy']['mae'] < 1.0
        ])

    def calculate_overall_score(self, test_results: Dict[str, Any]) -> int:
        """Calculate overall score (number of dimensions passed)."""
        dimensions = [
            test_results['specification'],
            test_results['implementation'],
            test_results['unit_tests']['coverage'] >= 85,
            test_results['determinism'],
            test_results['documentation'],
            test_results['compliance'] and test_results['security'],
            test_results['deployment'],
            self.check_exit_bar(test_results),
            test_results['integration'],
            test_results['business_impact'],
            test_results['operations'],
            test_results['continuous_improvement']
        ]

        return sum(dimensions)


if __name__ == "__main__":
    generator = ValidationReportGenerator(agent_id="GL-002", version="1.0.0")

    # Load test results
    with open("test_results.json", "r") as f:
        test_results = json.load(f)

    # Load benchmark results
    with open("benchmark_results.json", "r") as f:
        benchmark_results = json.load(f)

    # Generate report
    report = generator.generate_report(test_results, benchmark_results)

    # Save report
    with open(f"docs/certification/{generator.agent_id}_v{generator.version}_validation_report.md", "w") as f:
        f.write(report)

    print(f"Validation report generated: {generator.agent_id}_v{generator.version}_validation_report.md")
```

---

## Integration with Existing Validation Reports

### Link to Existing Reports

The evaluation pipeline automatically links to existing validation reports in the `validation_reports/` directory:

```python
# scripts/link_validation_reports.py

import os
import glob
from typing import List


def find_validation_reports(agent_id: str) -> List[str]:
    """Find all validation reports for an agent."""
    pattern = f"validation_reports/**/{agent_id}*.md"
    reports = glob.glob(pattern, recursive=True)
    return reports


def generate_report_index(agent_id: str):
    """Generate index of all validation reports for an agent."""
    reports = find_validation_reports(agent_id)

    index = f"# Validation Reports: {agent_id}\n\n"

    for report in reports:
        report_name = os.path.basename(report)
        index += f"- [{report_name}]({report})\n"

    return index


if __name__ == "__main__":
    agent_id = "GL-002"
    index = generate_report_index(agent_id)
    print(index)
```

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-03 | GL-TestEngineer | Initial evaluation pipeline |

---

**END OF DOCUMENT**
