# 🤖 GL-CSRD-APP: Agent Orchestration Guide

**Comprehensive Guide to AI Agent Integration & Automation**

**Version:** 1.0.0
**Date:** October 18, 2025
**Document Type:** Technical Guide
**Status:** Active Development

---

## 📋 TABLE OF CONTENTS

1. [Overview](#overview)
2. [Agent Ecosystem](#agent-ecosystem)
3. [Orchestration Strategies](#orchestration-strategies)
4. [Workflow Patterns](#workflow-patterns)
5. [Configuration Guide](#configuration-guide)
6. [Implementation Examples](#implementation-examples)
7. [Monitoring & Observability](#monitoring--observability)
8. [Troubleshooting](#troubleshooting)

---

## 1. Overview

### **What is Agent Orchestration?**

Agent orchestration is the automated coordination of multiple specialized AI agents to perform complex workflows. In GL-CSRD-APP, we orchestrate **18 specialized agents** (14 GreenLang platform agents + 4 CSRD-specific domain agents) to ensure:

1. **Quality Assurance**: Code quality, type safety, dependency management
2. **Security Compliance**: Vulnerability scanning, secrets detection, supply chain security
3. **Data Integrity**: ESG data lineage, PII protection, reproducibility
4. **Regulatory Compliance**: CSRD directive compliance, ESRS standards, XBRL validation
5. **Production Readiness**: Exit bar validation, deployment gates

### **Benefits of Orchestration**

**Automated Quality Gates:**
- No manual code review bottlenecks
- Consistent quality enforcement
- Immediate feedback to developers

**Comprehensive Validation:**
- Multi-layer validation (code → data → compliance → production)
- Zero blind spots in quality assurance
- Regulatory confidence

**Accelerated Development:**
- Parallel agent execution
- Automated workflows
- Faster time-to-production

**Audit Trail:**
- Complete agent execution history
- Decision provenance
- Regulatory compliance documentation

---

## 2. Agent Ecosystem

### **18-Agent Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│             GREENLANG PLATFORM AGENTS (14)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  QUALITY & CODE REVIEW                                          │
│  ├── GL-CodeSentinel (Red)     - Code quality & linting        │
│  ├── GL-PackQC (Cyan)          - Pack quality assurance        │
│  └── Greenlang-TaskChecker (Blue) - Task completion verification│
│                                                                 │
│  SECURITY & VALIDATION                                          │
│  ├── GL-SecScan (Purple)       - Security vulnerability scan   │
│  ├── GL-ConnectorValidator (Orange) - Connector validation     │
│  ├── GL-PolicyLinter (Green)   - OPA policy auditing          │
│  └── GL-SupplyChainSentinel (Yellow) - SBOM & signatures      │
│                                                                 │
│  DATA & FLOW VALIDATION                                         │
│  ├── GL-DataFlowGuardian (Teal) - Data lineage & PII          │
│  └── GL-DeterminismAuditor (Blue) - Reproducibility checks    │
│                                                                 │
│  SPECIFICATION & COMPLIANCE                                     │
│  ├── GL-SpecGuardian (Default) - Spec v1.0 compliance         │
│  ├── GL-ExitBarAuditor (Red)  - Production readiness          │
│  └── GL-HubRegistrar (Indigo) - Registry standards            │
│                                                                 │
│  DOCUMENTATION & TRACKING                                       │
│  ├── Product-DevelopmentTracker (Cyan) - Progress tracking    │
│  └── ProjectStatusReporter (Blue) - Stakeholder reports       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│           CSRD DOMAIN AGENTS (4 - To Be Created)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  REGULATORY COMPLIANCE                                          │
│  ├── GL-CSRDCompliance (Green) - CSRD directive compliance     │
│  ├── GL-SustainabilityMetrics (Teal) - ESG KPI quality        │
│  ├── GL-SupplyChainCSRD (Orange) - Value chain transparency   │
│  └── GL-XBRLValidator (Blue)   - ESEF technical compliance    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### **Agent Capabilities Matrix**

| Agent | Domain | Enforcement | Output | Invocation Trigger |
|-------|--------|-------------|--------|-------------------|
| **GL-CodeSentinel** | Code Quality | High (FAIL on issues) | Structured JSON | Code commits |
| **GL-SecScan** | Security | Critical (BLOCKER) | Security Report | Code commits, releases |
| **GL-DataFlowGuardian** | Data Integrity | Critical | Flow Analysis JSON | Pipeline execution |
| **GL-DeterminismAuditor** | Reproducibility | Critical | PASS/FAIL + Analysis | Post-calculation |
| **GL-ExitBarAuditor** | Production Readiness | Critical (GO/NO_GO) | Executive Decision | Release preparation |
| **GL-CSRDCompliance** | CSRD Compliance | Critical | Compliance Report | Report generation |
| **GL-SustainabilityMetrics** | ESG Quality | High | Data Quality Report | Post-calculation |
| **GL-SupplyChainCSRD** | Value Chain | High | Supply Chain Risks | ESRS S2 material |
| **GL-XBRLValidator** | XBRL/ESEF | Critical | Technical Validation | Report generation |

---

## 3. Orchestration Strategies

### **Strategy 1: Development Quality Workflow**

**Trigger:** Code commit to CSRD repository

**Objective:** Ensure code quality before merge

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEVELOPMENT QUALITY WORKFLOW                  │
└─────────────────────────────────────────────────────────────────┘

Developer commits code
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: Code Quality (GL-CodeSentinel)                         │
│                                                                  │
│ Checks:                                                          │
│  ✓ Lint errors (flake8, ruff)                                   │
│  ✓ Type errors (mypy)                                           │
│  ✓ Import graph analysis                                        │
│  ✓ Circular dependencies                                        │
│  ✓ Portability (OS-independent paths)                           │
│  ✓ CLI usability (help text < 24-30 lines)                      │
│  ✓ Dangerous patterns (SQL injection, resource leaks)           │
│                                                                  │
│ Failure Criteria:                                               │
│  • ANY lint error → FAIL                                        │
│  • ANY type error → FAIL                                        │
│  • Circular dependency → FAIL                                   │
│                                                                  │
│ Output: Structured JSON with issue categories                   │
└─────────────────────────────────────────────────────────────────┘
  ↓ PASS
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: Security Scan (GL-SecScan)                             │
│                                                                  │
│ Checks:                                                          │
│  ✓ Secrets detection (API keys, credentials, tokens)            │
│  ✓ Policy bypass (direct HTTP calls without wrappers)           │
│  ✓ Dependency vulnerabilities (CVE scanning)                    │
│  ✓ Input validation                                             │
│  ✓ Authentication/authorization bypass                          │
│                                                                  │
│ Severity Framework:                                             │
│  • BLOCKER: Hardcoded secrets, critical CVEs (≥9.0)             │
│  • WARN: High CVEs (7.0-8.9), deprecated practices              │
│                                                                  │
│ Failure Criteria:                                               │
│  • ANY BLOCKER finding → FAIL                                   │
│  • Secrets in code → FAIL                                       │
│  • >0 critical CVEs → FAIL                                      │
│  • >3 high CVEs → FAIL                                          │
│                                                                  │
│ Output: PASSED/FAILED + findings + exact fixes                  │
└─────────────────────────────────────────────────────────────────┘
  ↓ PASS
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: Spec Compliance (GL-SpecGuardian) [If manifest files]  │
│                                                                  │
│ Checks (if pack.yaml, gl.yaml, run.json modified):              │
│  ✓ pack.yaml validation (dependencies, metadata)                │
│  ✓ gl.yaml validation (configuration)                           │
│  ✓ run.json validation (execution params)                       │
│  ✓ Breaking change detection                                    │
│                                                                  │
│ Failure Criteria:                                               │
│  • Missing required fields → FAIL                               │
│  • Type mismatches → FAIL                                       │
│  • Breaking changes without migration notes → FAIL              │
│                                                                  │
│ Output: JSON with errors, warnings, autofix suggestions         │
└─────────────────────────────────────────────────────────────────┘
  ↓ PASS
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: Task Verification (Greenlang-TaskChecker)              │
│                                                                  │
│ Checks:                                                          │
│  ✓ Functional completeness                                      │
│  ✓ Edge case handling                                           │
│  ✓ Error handling validation                                    │
│  ✓ Gap identification with severity                             │
│                                                                  │
│ Output: Completion %, missing items, prioritized to-do list     │
└─────────────────────────────────────────────────────────────────┘
  ↓ PASS
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 5: Progress Documentation (Product-DevelopmentTracker)    │
│                                                                  │
│ Actions:                                                         │
│  • Document changes with exact timestamp                        │
│  • Update activity log with category (Feature, Bug Fix, etc.)   │
│  • Link related changes                                         │
│                                                                  │
│ Output: Chronological activity record                           │
└─────────────────────────────────────────────────────────────────┘
  ↓
✅ MERGE APPROVED
```

**Implementation (.github/workflows/csrd_quality.yml):**

```yaml
name: CSRD Development Quality

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  quality-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Stage 1 - Code Quality (GL-CodeSentinel)
        run: |
          python -m claude_code.agents.gl_codesentinel \
            --path . \
            --output quality_report.json

      - name: Check Code Quality Results
        run: |
          if [ $(jq '.status' quality_report.json) == "FAIL" ]; then
            echo "❌ Code quality checks failed"
            jq '.issues' quality_report.json
            exit 1
          fi

      - name: Stage 2 - Security Scan (GL-SecScan)
        run: |
          python -m claude_code.agents.gl_secscan \
            --path . \
            --output security_report.json

      - name: Check Security Results
        run: |
          if [ $(jq '.status' security_report.json) == "FAIL" ]; then
            echo "❌ Security scan failed"
            jq '.findings' security_report.json
            exit 1
          fi

      - name: Stage 3 - Spec Compliance (GL-SpecGuardian)
        if: contains(github.event.head_commit.modified, 'pack.yaml') || contains(github.event.head_commit.modified, 'gl.yaml')
        run: |
          python -m claude_code.agents.gl_spec_guardian \
            --path . \
            --output spec_report.json

      - name: Stage 4 - Task Verification (Greenlang-TaskChecker)
        run: |
          python -m claude_code.agents.greenlang_task_checker \
            --task-description "${{ github.event.head_commit.message }}" \
            --output task_report.json

      - name: Stage 5 - Update Development Tracker
        run: |
          python -m claude_code.agents.product_development_tracker \
            --action update \
            --changes "${{ github.event.head_commit.message }}" \
            --timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

      - name: Upload Reports
        uses: actions/upload-artifact@v3
        with:
          name: quality-reports
          path: |
            quality_report.json
            security_report.json
            spec_report.json
            task_report.json
```

---

### **Strategy 2: Data Pipeline Validation Workflow**

**Trigger:** CSRD pipeline execution

**Objective:** Ensure data integrity and compliance

```
┌─────────────────────────────────────────────────────────────────┐
│                 DATA PIPELINE VALIDATION WORKFLOW                │
└─────────────────────────────────────────────────────────────────┘

Pipeline execution started
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ PRE-EXECUTION: Data Flow Analysis (GL-DataFlowGuardian)         │
│                                                                  │
│ Checks:                                                          │
│  ✓ ESG data lineage tracking (source → sink)                    │
│  ✓ PII detection (workforce data, employee info)                │
│  ✓ GDPR compliance validation                                   │
│  ✓ Data residency validation                                    │
│  ✓ Access control verification                                  │
│  ✓ Schema evolution compatibility                               │
│                                                                  │
│ Failure Conditions:                                             │
│  • Data loss detected → FAIL                                    │
│  • PII exposure → FAIL                                          │
│  • Missing error handling → FAIL                                │
│  • Circular dependencies → FAIL                                 │
│  • Access violations → FAIL                                     │
│                                                                  │
│ Output: Data flow map + integrity checks + compliance matrix    │
└─────────────────────────────────────────────────────────────────┘
  ↓ PASS
┌─────────────────────────────────────────────────────────────────┐
│ PRE-EXECUTION: Policy Validation (GL-PolicyLinter) [If OPA used]│
│                                                                  │
│ Checks:                                                          │
│  ✓ Egress allowlist verification (no wildcards)                 │
│  ✓ Data residency compliance                                    │
│  ✓ Enterprise Framework vintage (≥2024)                         │
│  ✓ License allowlist (no GPL/copyleft)                          │
│  ✓ Migration readiness (learning → deny-by-default)             │
│                                                                  │
│ Failure Triggers:                                               │
│  • Non-allowlisted egress → FAIL                                │
│  • Missing residency validation → FAIL                          │
│  • EF<2024 → FAIL                                               │
│  • GPL licenses → FAIL                                          │
│  • Default 'allow' rules → FAIL                                 │
│                                                                  │
│ Output: Critical violations + migration checklist               │
└─────────────────────────────────────────────────────────────────┘
  ↓ PASS
┌─────────────────────────────────────────────────────────────────┐
│ PRE-EXECUTION: Connector Validation (GL-ConnectorValidator)     │
│              [If ERP connectors used]                            │
│                                                                  │
│ Checks:                                                          │
│  ✓ Authentication security (no hardcoded credentials)           │
│  ✓ Rate limiting & throttling (backoff, connection limits)      │
│  ✓ Error handling & resilience (retry, circuit breaker, timeouts)│
│  ✓ Data validation & transformation (sanitization, schema)      │
│  ✓ Performance & resource mgmt (connection pooling, memory)     │
│                                                                  │
│ Failure Criteria:                                               │
│  • Hardcoded credentials → FAIL                                 │
│  • No retry logic → FAIL                                        │
│  • Missing rate limits → FAIL                                   │
│  • No timeouts → FAIL                                           │
│  • Memory leaks → FAIL                                          │
│  • Thread safety issues → FAIL                                  │
│                                                                  │
│ Output: production_ready (true/false) + recommendations         │
└─────────────────────────────────────────────────────────────────┘
  ↓ PASS
┌─────────────────────────────────────────────────────────────────┐
│ PIPELINE EXECUTION: CSRD 6-Agent Pipeline                        │
│                                                                  │
│  Intake → Materiality → Calculate → Aggregate → Report → Audit  │
│                                                                  │
│ [Standard CSRD pipeline execution - see main architecture docs]  │
└─────────────────────────────────────────────────────────────────┘
  ↓ COMPLETE
┌─────────────────────────────────────────────────────────────────┐
│ POST-EXECUTION: Reproducibility Check (GL-DeterminismAuditor)   │
│                                                                  │
│ Checks:                                                          │
│  ✓ Hash comparison (Run A vs Run B)                             │
│  ✓ Local vs K8s environment comparison                          │
│  ✓ Non-determinism root cause analysis                          │
│  ✓ Quantization and seed validation                             │
│  ✓ Library version mismatch detection                           │
│                                                                  │
│ Process:                                                         │
│  1. Re-run pipeline with same inputs                            │
│  2. Compare output hashes (SHA-256)                             │
│  3. If ANY mismatch → FAIL + root cause analysis                │
│  4. Identify source: floating-point, temporal, path ordering    │
│                                                                  │
│ Failure Criteria:                                               │
│  • ANY hash mismatch = FAIL (even one difference)               │
│                                                                  │
│ Output: PASS/FAIL + hash mismatches + root causes + fixes       │
└─────────────────────────────────────────────────────────────────┘
  ↓ PASS
✅ PIPELINE VALIDATED
```

**Implementation (scripts/validate_pipeline.py):**

```python
#!/usr/bin/env python3
"""
Data Pipeline Validation with Agent Orchestration
"""

import sys
from claude_code.agents import (
    GLDataFlowGuardian,
    GLPolicyLinter,
    GLConnectorValidator,
    GLDeterminismAuditor
)
from csrd_pipeline import CSRDPipeline

def validate_pipeline(config_path, esg_data, company_profile, materiality):
    """Execute data pipeline validation workflow"""

    print("=" * 70)
    print("DATA PIPELINE VALIDATION WORKFLOW")
    print("=" * 70)

    # STAGE 1: Data Flow Guardian
    print("\n[1/4] Data Flow Analysis (GL-DataFlowGuardian)...")
    data_flow_guardian = GLDataFlowGuardian()
    data_flow_result = data_flow_guardian.validate(
        pipeline_config=config_path,
        data_sources=[esg_data, company_profile]
    )

    if not data_flow_result['passed']:
        print(f"❌ Data flow validation FAILED")
        print(f"   Critical issues: {len(data_flow_result['critical_issues'])}")
        for issue in data_flow_result['critical_issues']:
            print(f"   - {issue['description']}")
        return False

    print(f"✅ Data flow validation PASSED")
    print(f"   Data sources: {len(data_flow_result['data_sources'])}")
    print(f"   PII fields: {len(data_flow_result['pii_fields'])}")

    # STAGE 2: Policy Linter (if OPA policies exist)
    print("\n[2/4] Policy Validation (GL-PolicyLinter)...")
    policy_linter = GLPolicyLinter()
    policy_result = policy_linter.audit(policy_dir="policies/")

    if not policy_result['passed']:
        print(f"❌ Policy validation FAILED")
        for violation in policy_result['critical_violations']:
            print(f"   - {violation}")
        return False

    print(f"✅ Policy validation PASSED")

    # STAGE 3: Connector Validator (if connectors used)
    print("\n[3/4] Connector Validation (GL-ConnectorValidator)...")
    connector_validator = GLConnectorValidator()
    connector_result = connector_validator.validate(connectors_dir="connectors/")

    if not connector_result['production_ready']:
        print(f"❌ Connector validation FAILED")
        for issue in connector_result['critical_issues']:
            print(f"   - {issue}")
        return False

    print(f"✅ Connector validation PASSED")

    # STAGE 4: Run pipeline
    print("\n[4/4] Executing CSRD pipeline...")
    pipeline = CSRDPipeline(config_path=config_path)

    # Run 1
    result1 = pipeline.run(
        esg_data_file=esg_data,
        company_profile=company_profile,
        materiality_assessment=materiality,
        output_path="output/run1_report.zip"
    )

    # Run 2 (reproducibility check)
    result2 = pipeline.run(
        esg_data_file=esg_data,
        company_profile=company_profile,
        materiality_assessment=materiality,
        output_path="output/run2_report.zip"
    )

    # STAGE 5: Determinism Auditor
    print("\n[5/4] Reproducibility Check (GL-DeterminismAuditor)...")
    determinism_auditor = GLDeterminismAuditor()
    determinism_result = determinism_auditor.verify(
        run_a_output=result1,
        run_b_output=result2
    )

    if not determinism_result['passed']:
        print(f"❌ Reproducibility check FAILED")
        print(f"   Hash mismatches: {len(determinism_result['mismatches'])}")
        for mismatch in determinism_result['mismatches']:
            print(f"   - {mismatch['metric']}: {mismatch['root_cause']}")
        return False

    print(f"✅ Reproducibility check PASSED")
    print(f"   All hashes identical: {determinism_result['identical_hashes']}/{determinism_result['total_hashes']}")

    print("\n" + "=" * 70)
    print("✅ DATA PIPELINE VALIDATION COMPLETE")
    print("=" * 70)
    return True

if __name__ == '__main__':
    success = validate_pipeline(
        config_path='config/csrd_config.yaml',
        esg_data='examples/demo_esg_data.csv',
        company_profile='examples/demo_company_profile.json',
        materiality='examples/demo_materiality.json'
    )

    sys.exit(0 if success else 1)
```

---

### **Strategy 3: Release Readiness Workflow**

**Trigger:** Version tagged for release

**Objective:** Ensure production readiness

```
┌─────────────────────────────────────────────────────────────────┐
│                   RELEASE READINESS WORKFLOW                     │
└─────────────────────────────────────────────────────────────────┘

Release candidate tagged (e.g., v1.0.0)
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: Pack Quality (GL-PackQC)                               │
│                                                                  │
│ Checks:                                                          │
│  ✓ Dependency resolution (no conflicts)                         │
│  ✓ Resource optimization (<100MB, warn >50MB)                   │
│  ✓ Metadata completeness (name, version, author, license)       │
│  ✓ Documentation (README, API docs, migration guides)           │
│  ✓ Version compatibility testing                                │
│                                                                  │
│ Quality Score: 0-100                                            │
│  • Dependencies: 25%                                            │
│  • Resources: 20%                                               │
│  • Metadata: 20%                                                │
│  • Documentation: 15%                                           │
│  • Tests: 10%                                                   │
│  • Versioning: 10%                                              │
│                                                                  │
│ Output: Quality score + critical issues + recommendations       │
└─────────────────────────────────────────────────────────────────┘
  ↓ PASS
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: Supply Chain Security (GL-SupplyChainSentinel)         │
│                                                                  │
│ Checks:                                                          │
│  ✓ SBOM validation (SPDX 2.2+ compliance)                       │
│  ✓ Cosign signature verification (keyless signing)              │
│  ✓ Provenance validation (build timestamp, builder, source)     │
│  ✓ OIDC identity verification                                   │
│  ✓ Certificate expiration check                                 │
│                                                                  │
│ Enforcement: ANY FAIL = OVERALL FAIL (strict)                   │
│                                                                  │
│ Failure Triggers:                                               │
│  • Unsigned artifacts → FAIL                                    │
│  • Signature mismatch → FAIL                                    │
│  • Unverifiable OIDC identity → FAIL                            │
│  • Expired certificates → FAIL                                  │
│  • Invalid SPDX format → FAIL                                   │
│                                                                  │
│ Output: PASS/FAIL + component-level results + remediation steps │
└─────────────────────────────────────────────────────────────────┘
  ↓ PASS
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: Exit Bar Validation (GL-ExitBarAuditor)                │
│                                                                  │
│ MUST Criteria (Binary Pass/Fail):                               │
│  ✓ Code coverage ≥80%                                           │
│  ✓ Zero critical bugs                                           │
│  ✓ All tests passing                                            │
│  ✓ Zero critical CVEs                                           │
│  ✓ Security scans passed                                        │
│  ✓ SBOM signed                                                  │
│  ✓ Monitoring ready                                             │
│  ✓ Rollback plan exists                                         │
│                                                                  │
│ SHOULD Criteria (80% threshold for readiness):                  │
│  ✓ No memory leaks                                              │
│  ✓ SLA compliance verified                                      │
│  ✓ Runbooks complete                                            │
│  ✓ Feature flags configured                                     │
│  ✓ Change approval obtained                                     │
│  ✓ Risk assessment complete                                     │
│  ✓ GDPR/SOC2 checks passed                                      │
│                                                                  │
│ NO_GO Triggers:                                                 │
│  • Critical vulnerabilities → NO_GO                             │
│  • Failed tests → NO_GO                                         │
│  • Missing rollback plan → NO_GO                                │
│  • No change approval → NO_GO                                   │
│                                                                  │
│ Output: GO/NO_GO + readiness score (0-100) + blocking issues    │
└─────────────────────────────────────────────────────────────────┘
  ↓ GO
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: Hub Registry Validation (GL-HubRegistrar) [If publishing]│
│                                                                  │
│ Checks:                                                          │
│  ✓ Package naming (no typosquatting, reserved words)            │
│  ✓ Versioning (semantic versioning, no conflicts)               │
│  ✓ License compatibility (valid license file)                   │
│  ✓ Documentation (README, API docs, migration guides)           │
│  ✓ Security vulnerability scan                                  │
│  ✓ Malicious pattern detection                                  │
│  ✓ No sensitive data (secrets, credentials)                     │
│                                                                  │
│ Review Triggers (Manual Review Required):                       │
│  • First-time publisher                                         │
│  • Significant version jump (1.0 → 2.0)                         │
│  • License change                                               │
│  • Ownership transfer                                           │
│  • Unusual patterns (obfuscation, external calls)               │
│  • Package size >50MB                                           │
│                                                                  │
│ Output: APPROVED/REJECTED/NEEDS_REVIEW + validation results     │
└─────────────────────────────────────────────────────────────────┘
  ↓ APPROVED
✅ RELEASE APPROVED FOR PRODUCTION
```

**Implementation (.github/workflows/csrd_release.yml):**

```yaml
name: CSRD Release Workflow

on:
  push:
    tags:
      - 'v*'

jobs:
  release-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Stage 1 - Pack Quality (GL-PackQC)
        run: |
          python -m claude_code.agents.gl_packqc \
            --path . \
            --output packqc_report.json

          # Check quality score
          QUALITY_SCORE=$(jq '.quality_score' packqc_report.json)
          if [ $QUALITY_SCORE -lt 80 ]; then
            echo "❌ Quality score too low: $QUALITY_SCORE (minimum: 80)"
            exit 1
          fi

      - name: Stage 2 - Supply Chain Security (GL-SupplyChainSentinel)
        run: |
          # Generate SBOM
          python -m greenlang.sbom.generate --format spdx --output sbom.spdx.json

          # Sign with Cosign (keyless)
          cosign sign-blob --yes sbom.spdx.json > sbom.sig

          # Validate
          python -m claude_code.agents.gl_supply_chain_sentinel \
            --sbom sbom.spdx.json \
            --signature sbom.sig \
            --output supply_chain_report.json

          # Check result
          if [ $(jq '.status' supply_chain_report.json) != "PASS" ]; then
            echo "❌ Supply chain validation failed"
            jq '.violations' supply_chain_report.json
            exit 1
          fi

      - name: Stage 3 - Exit Bar Validation (GL-ExitBarAuditor)
        run: |
          python -m claude_code.agents.gl_exitbar_auditor \
            --path . \
            --tests-dir tests/ \
            --coverage-report coverage.xml \
            --output exitbar_report.json

          # Check GO/NO_GO
          VERDICT=$(jq -r '.verdict' exitbar_report.json)
          if [ "$VERDICT" != "GO" ]; then
            echo "❌ Exit bar validation: NO_GO"
            jq '.blocking_issues' exitbar_report.json
            exit 1
          fi

          echo "✅ Exit bar validation: GO"
          echo "   Readiness score: $(jq '.readiness_score' exitbar_report.json)"

      - name: Stage 4 - Hub Registry Validation (GL-HubRegistrar)
        if: github.event_name == 'release'
        run: |
          python -m claude_code.agents.gl_hub_registrar \
            --package-path . \
            --output registry_report.json

          STATUS=$(jq -r '.status' registry_report.json)
          if [ "$STATUS" == "REJECTED" ]; then
            echo "❌ Registry validation rejected"
            jq '.blocking_issues' registry_report.json
            exit 1
          elif [ "$STATUS" == "NEEDS_REVIEW" ]; then
            echo "⚠️  Manual review required"
            jq '.review_triggers' registry_report.json
          else
            echo "✅ Registry validation approved"
          fi

      - name: Create Release
        if: success()
        uses: actions/create-release@v1
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
          draft: false
          prerelease: false
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Upload Release Assets
        uses: actions/upload-release-asset@v1
        with:
          upload_url: ${{ steps.create_release.outputs.upload_url }}
          asset_path: ./csrd-app.zip
          asset_name: csrd-app-${{ github.ref }}.zip
          asset_content_type: application/zip
```

---

### **Strategy 4: CSRD Domain Validation Workflow**

**Trigger:** CSRD report generation complete

**Objective:** Comprehensive compliance validation

```
┌─────────────────────────────────────────────────────────────────┐
│                CSRD DOMAIN VALIDATION WORKFLOW                   │
└─────────────────────────────────────────────────────────────────┘

CSRD report generated (6-agent pipeline complete)
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ DOMAIN AGENT 1: CSRD Compliance (GL-CSRDCompliance)             │
│                                                                  │
│ Checks:                                                          │
│  ✓ Double materiality assessment completeness                   │
│  ✓ ESRS disclosure completeness (all material topics)           │
│  ✓ Timeline compliance (Phase 1-4 deadlines)                    │
│  ✓ Subsidiary consolidation validation                          │
│  ✓ External assurance readiness                                 │
│  ✓ Basis of preparation documentation                           │
│  ✓ Cross-reference completeness                                 │
│                                                                  │
│ Failure Criteria:                                               │
│  • Missing materiality assessment → FAIL                        │
│  • Incomplete ESRS disclosures for material topics → FAIL       │
│  • Timeline violations (e.g., Phase 1 report due Q1 2025) → FAIL│
│  • Consolidation errors (subsidiaries not aggregated) → FAIL    │
│                                                                  │
│ Output: PASS/FAIL + critical violations + remediation steps     │
└─────────────────────────────────────────────────────────────────┘
  ↓ PASS
┌─────────────────────────────────────────────────────────────────┐
│ DOMAIN AGENT 2: Sustainability Metrics (GL-SustainabilityMetrics)│
│                                                                  │
│ Checks:                                                          │
│  ✓ Scope 1/2/3 emissions validation (GHG Protocol compliance)   │
│  ✓ Energy consumption verification (total = sum of sources)     │
│  ✓ Water metrics validation (withdrawal ≥ consumption)          │
│  ✓ Waste metrics validation (generated ≥ recycled + landfilled) │
│  ✓ Social metrics verification (employee counts, safety rates)  │
│  ✓ Governance metrics validation (board composition, ethics)    │
│  ✓ Year-over-year consistency (trend validation)                │
│  ✓ Benchmark deviation analysis (flagged if >2 std dev)         │
│                                                                  │
│ Failure Criteria:                                               │
│  • Calculation errors (formulas incorrectly applied) → FAIL     │
│  • Inconsistent methodology (different vs. prior year) → FAIL   │
│  • Missing baseline data (no historical comparison) → FAIL      │
│  • Statistical outliers without explanation → WARNING           │
│                                                                  │
│ Output: PASS/FAIL + data quality issues + outlier flags         │
└─────────────────────────────────────────────────────────────────┘
  ↓ PASS
┌─────────────────────────────────────────────────────────────────┐
│ DOMAIN AGENT 3: Supply Chain CSRD (GL-SupplyChainCSRD)          │
│              [Only if ESRS S2 is material]                       │
│                                                                  │
│ Checks:                                                          │
│  ✓ Supplier ESG assessment coverage (% of spend covered)        │
│  ✓ Conflict minerals tracking (3TG declarations)                │
│  ✓ Labor compliance verification (ILO conventions)              │
│  ✓ Tier 2/3 supplier visibility (indirect suppliers)            │
│  ✓ Supply chain carbon footprint (Scope 3 Category 1)           │
│  ✓ High-risk supplier identification (countries, industries)    │
│                                                                  │
│ Failure Criteria:                                               │
│  • High-risk suppliers without ESG assessments → FAIL           │
│  • Missing conflict mineral declarations → FAIL                 │
│  • Unverified labor practices (no audits) → FAIL                │
│  • Incomplete supply chain mapping (major gaps) → WARNING       │
│                                                                  │
│ Output: PASS/FAIL + supplier risks + coverage gaps              │
└─────────────────────────────────────────────────────────────────┘
  ↓ PASS
┌─────────────────────────────────────────────────────────────────┐
│ DOMAIN AGENT 4: XBRL Validator (GL-XBRLValidator)               │
│                                                                  │
│ Checks:                                                          │
│  ✓ XBRL taxonomy validation (ESRS 2024 taxonomy)                │
│  ✓ iXBRL rendering verification (human-readable + machine-readable)│
│  ✓ ESEF package completeness (all required files)               │
│  ✓ Digital signature validation (if required)                   │
│  ✓ EU portal submission readiness (format compliance)           │
│  ✓ Tag usage correctness (1,000+ ESRS tags)                     │
│  ✓ Context and dimension validation                             │
│                                                                  │
│ Failure Criteria:                                               │
│  • Invalid XBRL tags (wrong taxonomy) → FAIL                    │
│  • Missing iXBRL elements → FAIL                                │
│  • ESEF package errors (incorrect structure) → FAIL             │
│  • Unsigned digital report (if required) → FAIL                 │
│  • Invalid contexts/dimensions → FAIL                           │
│                                                                  │
│ Output: PASS/FAIL + technical errors + ESEF readiness           │
└─────────────────────────────────────────────────────────────────┘
  ↓ PASS
✅ CSRD DOMAIN VALIDATION COMPLETE
   Report is compliant and submission-ready
```

**Implementation (scripts/validate_csrd_report.py):**

```python
#!/usr/bin/env python3
"""
CSRD Domain Validation Workflow
"""

import sys
from claude_code.agents import (
    GLCSRDCompliance,
    GLSustainabilityMetrics,
    GLSupplyChainCSRD,
    GLXBRLValidator
)

def validate_csrd_report(report_package_path, materiality_matrix):
    """Execute CSRD domain validation workflow"""

    print("=" * 70)
    print("CSRD DOMAIN VALIDATION WORKFLOW")
    print("=" * 70)

    # Extract report package
    report = extract_report_package(report_package_path)

    # AGENT 1: CSRD Compliance
    print("\n[1/4] CSRD Compliance Validation (GL-CSRDCompliance)...")
    csrd_compliance = GLCSRDCompliance()
    compliance_result = csrd_compliance.validate(
        report=report,
        materiality_matrix=materiality_matrix
    )

    if not compliance_result['passed']:
        print(f"❌ CSRD compliance validation FAILED")
        print(f"   Critical violations: {len(compliance_result['critical_violations'])}")
        for violation in compliance_result['critical_violations']:
            print(f"   - {violation['description']}")
        return False

    print(f"✅ CSRD compliance validation PASSED")
    print(f"   Disclosure completeness: {compliance_result['disclosure_completeness']}%")

    # AGENT 2: Sustainability Metrics
    print("\n[2/4] Sustainability Metrics Validation (GL-SustainabilityMetrics)...")
    sustainability_metrics = GLSustainabilityMetrics()
    metrics_result = sustainability_metrics.validate(
        report=report,
        historical_data=load_historical_data()
    )

    if not metrics_result['passed']:
        print(f"❌ Sustainability metrics validation FAILED")
        for issue in metrics_result['data_quality_issues']:
            print(f"   - {issue}")
        return False

    print(f"✅ Sustainability metrics validation PASSED")
    print(f"   Metrics validated: {metrics_result['metrics_validated']}")
    if metrics_result['outliers']:
        print(f"   ⚠️  Outliers flagged for review: {len(metrics_result['outliers'])}")

    # AGENT 3: Supply Chain CSRD (if ESRS S2 material)
    if is_s2_material(materiality_matrix):
        print("\n[3/4] Supply Chain Validation (GL-SupplyChainCSRD)...")
        supply_chain = GLSupplyChainCSRD()
        supply_chain_result = supply_chain.validate(
            report=report,
            supplier_data=load_supplier_data()
        )

        if not supply_chain_result['passed']:
            print(f"❌ Supply chain validation FAILED")
            for risk in supply_chain_result['supplier_risks']:
                print(f"   - {risk}")
            return False

        print(f"✅ Supply chain validation PASSED")
        print(f"   Supplier coverage: {supply_chain_result['supplier_coverage']}%")
    else:
        print("\n[3/4] Supply Chain Validation (SKIPPED - ESRS S2 not material)")

    # AGENT 4: XBRL Validator
    print("\n[4/4] XBRL/ESEF Validation (GL-XBRLValidator)...")
    xbrl_validator = GLXBRLValidator()
    xbrl_result = xbrl_validator.validate(
        report=report,
        taxonomy_version="ESRS-2024"
    )

    if not xbrl_result['passed']:
        print(f"❌ XBRL validation FAILED")
        for error in xbrl_result['technical_errors']:
            print(f"   - {error}")
        return False

    print(f"✅ XBRL validation PASSED")
    print(f"   Tags validated: {xbrl_result['tags_validated']}")
    print(f"   ESEF ready: {xbrl_result['esef_ready']}")

    print("\n" + "=" * 70)
    print("✅ CSRD DOMAIN VALIDATION COMPLETE")
    print("   Report is compliant and submission-ready")
    print("=" * 70)
    return True

if __name__ == '__main__':
    success = validate_csrd_report(
        report_package_path='output/csrd_report_package.zip',
        materiality_matrix='output/materiality_matrix.json'
    )

    sys.exit(0 if success else 1)
```

---

## 4. Workflow Patterns

### **Pattern 1: Sequential Validation**

**When to use:** Each stage depends on previous stage passing

```python
def sequential_validation(artifact):
    """Execute agents sequentially, stop on first failure"""
    agents = [Agent1(), Agent2(), Agent3(), Agent4()]

    for i, agent in enumerate(agents, 1):
        result = agent.validate(artifact)

        if not result.passed:
            print(f"❌ Validation failed at stage {i}")
            return False

    return True
```

### **Pattern 2: Parallel Validation**

**When to use:** Agents are independent, can run concurrently

```python
import concurrent.futures

def parallel_validation(artifact):
    """Execute agents in parallel for speed"""
    agents = [Agent1(), Agent2(), Agent3(), Agent4()]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(agent.validate, artifact) for agent in agents]

        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    # All must pass
    return all(result.passed for result in results)
```

### **Pattern 3: Conditional Execution**

**When to use:** Some agents only run under specific conditions

```python
def conditional_validation(artifact, config):
    """Execute agents conditionally based on configuration"""
    results = {}

    # Always run
    results['base'] = Agent1().validate(artifact)

    # Conditional
    if config.has_manifest_changes:
        results['spec'] = GL_SpecGuardian().validate(artifact)

    if config.has_opa_policies:
        results['policy'] = GL_PolicyLinter().validate(artifact)

    if config.esrs_s2_material:
        results['supply_chain'] = GL_SupplyChainCSRD().validate(artifact)

    return all(r.passed for r in results.values())
```

---

## 5. Configuration Guide

### **Agent Configuration Files**

**Location:** `.claude/agents/config/`

```yaml
# .claude/agents/config/csrd_orchestration.yaml

orchestration:
  # Development quality workflow
  development:
    enabled: true
    agents:
      - gl-codesentinel
      - gl-secscan
      - gl-spec-guardian
      - greenlang-task-checker
      - product-development-tracker
    execution: sequential  # or parallel
    fail_fast: true  # Stop on first failure

  # Data pipeline validation workflow
  data_pipeline:
    enabled: true
    agents:
      - gl-dataflow-guardian
      - gl-policy-linter
      - gl-connector-validator
      - gl-determinism-auditor
    execution: sequential

  # Release readiness workflow
  release:
    enabled: true
    agents:
      - gl-packqc
      - gl-supply-chain-sentinel
      - gl-exitbar-auditor
      - gl-hub-registrar
    execution: sequential
    quality_gates:
      pack_quality_min_score: 80
      exit_bar_readiness_min: 80

  # CSRD domain validation workflow
  csrd_domain:
    enabled: true
    agents:
      - gl-csrd-compliance
      - gl-sustainability-metrics
      - gl-supply-chain-csrd  # Conditional on ESRS S2
      - gl-xbrl-validator
    execution: sequential

# Agent-specific configurations
agents:
  gl-codesentinel:
    rules:
      - lint
      - type
      - import
      - portability
      - cli
      - dangerous
    severity: high  # FAIL on any issue

  gl-secscan:
    scanners:
      - secrets
      - vulnerabilities
      - policy_bypass
    severity:
      blocker_on:
        - hardcoded_secrets
        - critical_cve
      fail_threshold:
        critical: 0
        high: 3

  gl-exitbar-auditor:
    must_criteria:
      - code_coverage_80
      - zero_critical_bugs
      - all_tests_passing
      - zero_critical_cves
      - sbom_signed
    should_criteria:
      - no_memory_leaks
      - sla_compliance
      - runbooks_complete
    thresholds:
      should_criteria_pass_rate: 0.8
```

---

## 6. Implementation Examples

### **Example 1: Git Pre-Commit Hook**

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running CSRD development quality checks..."

# Stage 1: Code Quality
python -m claude_code.agents.gl_codesentinel --path . --output /tmp/codesentinel.json
if [ $? -ne 0 ]; then
    echo "❌ Code quality check failed. Fix issues before committing."
    cat /tmp/codesentinel.json | jq '.issues'
    exit 1
fi

# Stage 2: Security Scan
python -m claude_code.agents.gl_secscan --path . --output /tmp/secscan.json
if [ $? -ne 0 ]; then
    echo "❌ Security scan failed. Fix vulnerabilities before committing."
    cat /tmp/secscan.json | jq '.findings'
    exit 1
fi

echo "✅ All quality checks passed"
exit 0
```

### **Example 2: Automated Testing with Agents**

```python
# tests/test_with_agents.py

import pytest
from claude_code.agents import GL_DeterminismAuditor
from csrd_pipeline import CSRDPipeline

def test_pipeline_reproducibility():
    """Test pipeline reproducibility using GL-DeterminismAuditor"""

    pipeline = CSRDPipeline(config_path='config/csrd_config.yaml')

    # Run 1
    result1 = pipeline.run(
        esg_data_file='tests/fixtures/test_data.csv',
        company_profile='tests/fixtures/test_company.json',
        materiality_assessment='tests/fixtures/test_materiality.json',
        output_path='/tmp/run1_report.zip'
    )

    # Run 2
    result2 = pipeline.run(
        esg_data_file='tests/fixtures/test_data.csv',
        company_profile='tests/fixtures/test_company.json',
        materiality_assessment='tests/fixtures/test_materiality.json',
        output_path='/tmp/run2_report.zip'
    )

    # Use GL-DeterminismAuditor to verify
    auditor = GL_DeterminismAuditor()
    audit_result = auditor.verify(run_a_output=result1, run_b_output=result2)

    assert audit_result['passed'], f"Reproducibility check failed: {audit_result['mismatches']}"
    assert audit_result['identical_hashes'] == audit_result['total_hashes']
```

---

## 7. Monitoring & Observability

### **Agent Execution Metrics**

**Track for each agent:**
- Execution count
- Pass/fail rate
- Average execution time
- Failure reasons
- Resource usage

**Dashboard Example (Prometheus + Grafana):**

```yaml
# prometheus.yml

- job_name: 'csrd_agents'
  metrics_path: '/metrics'
  static_configs:
    - targets: ['localhost:9090']

# Metrics exposed:
# - agent_execution_total{agent_name="gl-codesentinel", status="pass|fail"}
# - agent_execution_duration_seconds{agent_name="gl-codesentinel"}
# - agent_failure_reasons{agent_name="gl-codesentinel", reason="lint_error|type_error"}
```

---

## 8. Troubleshooting

### **Common Issues**

**Issue 1: Agent execution fails with import error**

```
ModuleNotFoundError: No module named 'claude_code.agents'
```

**Solution:**
```bash
# Install Claude Code CLI
pip install claude-code

# Or set PYTHONPATH
export PYTHONPATH=/path/to/claude-code:$PYTHONPATH
```

**Issue 2: Agent returns FAIL but no clear reason**

**Solution:**
- Check agent output JSON for detailed findings
- Enable verbose logging: `--log-level debug`
- Review agent-specific documentation in `.claude/agents/`

**Issue 3: Performance degradation with many agents**

**Solution:**
- Use parallel execution where possible
- Cache agent results for unchanged artifacts
- Skip non-critical agents in development builds

---

**Document Status:** ✅ Complete
**Last Updated:** 2025-10-18
**Next Review:** After Week 2 completion (agent integration)

**Ready to orchestrate! 🤖**
